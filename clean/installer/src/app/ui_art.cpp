// SPDX-License-Identifier: MIT
// WrongGPU launcher window: game art (ui_shell.h). Read at run time from the player's own Steam cache
// (<Steam>\appcache\librarycache\<appid>\, directly there or one hashed folder deeper): the wide library_hero.jpg and the
// game's logo.png (the Library's tiles and featured banner, the game page's banner), header.jpg (a tile without a hero),
// the portrait library_600x900.jpg or library_capsule.jpg (the compact header, the game page without a hero, the Not
// supported rows). Decoded with WIC on a worker thread at the size they are drawn (times the window's scale) and kept in
// memory: the art of the games drawn most recently (kKeepArt), the banner-sized hero of a few (kKeepBig), the others
// decoded again when they come back into view. A banner's hero is decoded before the tiles' art (it is the biggest thing on
// the page). When the window's scale changes (another monitor) the art decoded at the old scale is still drawn, scaled,
// until the new decode of each arrives, so nothing drops back to a plate. Nothing is copied, written or shipped; games
// without art (Epic, GOG, added by hand) get a generated plate. Written for this project.
#include "app/ui_shell.h"

#include <condition_variable>
#include <functional>
#include <deque>
#include <set>
#include <thread>

#include "wg_core/detect.h"

namespace wg::app::lui {

namespace {

// The games' art kept decoded (about 2 MB a game): a big library scrolled end to end does not keep every game it has shown;
// the banner-sized heroes (3 to 7 MB each) of the last few banners drawn
constexpr size_t kKeepArt = 96, kKeepBig = 6;
// The largest sizes the art is drawn at (DIPs): a tile (2 per row at the smallest window), a logo (the banner's), a banner
constexpr float kTileArtW = 340, kLogoW = 300, kBannerW = 1080;

struct ArtState {
    std::mutex m;
    std::condition_variable cv;
    std::map<wstring, Art> done;            // appid -> decoded art
    std::map<wstring, ImageP> big;          // appid -> the banner-sized hero (null: none)
    std::map<wstring, unsigned long long> used, bigUsed;  // appid -> when it was last asked for (ArtFor's / ArtBig's count)
    std::map<wstring, Art> oldDone;         // decoded at the scale before (a DPI change): drawn until the new decode arrives
    std::map<wstring, ImageP> oldBig;
    std::map<wstring, bool> heroOnDisk;     // appid -> the cache has its library_hero.jpg (HeroOnDisk; kept over ArtScale)
    float scale = 1;                        // the window's DPI scale: art is decoded at its drawn size times this
    unsigned long long tick = 0;
    std::deque<wstring> queue;
    std::set<wstring> queued;
    HWND notify{}; UINT msg = 0;
    bool stop = false, running = false;
    std::thread worker;
    wstring cache;                          // <Steam>\appcache\librarycache
} st;

wstring LibraryCache() {
    wstring steam = RegString(HKEY_CURRENT_USER, L"Software\\Valve\\Steam", L"SteamPath");
    for (auto& c : steam) if (c == L'/') c = L'\\';
    return steam.empty() ? wstring() : Join(steam, L"appcache\\librarycache");
}

// <cache>\<appid>\<name>, or one folder deeper (Steam keeps hashed sub-folders); the newest when there are several.
wstring FindArt(const wstring& dir, const wchar_t* name) {
    wstring best;
    FILETIME bestTime{};
    auto consider = [&](const wstring& p) {
        WIN32_FILE_ATTRIBUTE_DATA a{};
        if (!GetFileAttributesExW(p.c_str(), GetFileExInfoStandard, &a) || (a.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) return;
        if (best.empty() || CompareFileTime(&a.ftLastWriteTime, &bestTime) > 0) { best = p; bestTime = a.ftLastWriteTime; }
    };
    consider(Join(dir, name));
    for (auto& sub : ListDirs(dir)) consider(Join(sub, name));
    return best;
}

// Decodes a JPEG / PNG to premultiplied BGRA, scaled down to at most maxW pixels wide (high-quality scaler).
ImageP Decode(IWICImagingFactory* wic, const wstring& path, UINT maxW) {
    if (path.empty() || !wic) return nullptr;
    IWICBitmapDecoder* dec = nullptr; IWICBitmapFrameDecode* fr = nullptr; IWICFormatConverter* conv = nullptr; IWICBitmapScaler* sc = nullptr;
    ImageP out;
    if (SUCCEEDED(wic->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec)) && SUCCEEDED(dec->GetFrame(0, &fr)) &&
        SUCCEEDED(wic->CreateFormatConverter(&conv)) &&
        SUCCEEDED(conv->Initialize(fr, GUID_WICPixelFormat32bppPBGRA, WICBitmapDitherTypeNone, nullptr, 0, WICBitmapPaletteTypeCustom))) {
        UINT w = 0, h = 0;
        conv->GetSize(&w, &h);
        IWICBitmapSource* src = conv;
        if (w > maxW && w > 0) {
            const UINT nw = maxW, nh = std::max<UINT>(1, (UINT)((double)h * maxW / w));
            if (SUCCEEDED(wic->CreateBitmapScaler(&sc)) && SUCCEEDED(sc->Initialize(conv, nw, nh, WICBitmapInterpolationModeHighQualityCubic))) { src = sc; w = nw; h = nh; }
        }
        if (w > 0 && h > 0 && w <= 8192 && h <= 8192) {
            auto img = std::make_shared<Image>();
            img->w = (int)w; img->h = (int)h;
            img->px.resize((size_t)w * h * 4);
            if (SUCCEEDED(src->CopyPixels(nullptr, w * 4, (UINT)img->px.size(), img->px.data()))) out = img;
        }
    }
    if (sc) sc->Release();
    if (conv) conv->Release();
    if (fr) fr->Release();
    if (dec) dec->Release();
    return out;
}

// Keeps the newest `keep` of a map by their last use (a quarter dropped at a time); they are decoded again when drawn again
template <class M> void Trim(M& m, std::map<wstring, unsigned long long>& used, size_t keep) {
    if (m.size() <= keep) return;
    std::vector<std::pair<unsigned long long, wstring>> age;
    for (auto& d : m) { const auto u = used.find(d.first); age.push_back({u != used.end() ? u->second : 0ULL, d.first}); }
    std::sort(age.begin(), age.end());
    for (size_t k = 0; k < std::max<size_t>(1, keep / 4) && k < age.size(); ++k) { m.erase(age[k].second); used.erase(age[k].second); }
}

UINT Px(float dips, float scale) { return (UINT)std::ceil(dips * std::max(1.f, scale)); }

static void Queue(const wstring& job) {  // under st.m; a banner's hero goes before the tiles' art
    if (!st.running || st.queued.count(job)) return;
    st.queued.insert(job);
    if (job.rfind(L"big:", 0) == 0) st.queue.push_front(job);
    else st.queue.push_back(job);
    st.cv.notify_one();
}

// `trim`: false for the offscreen renders (ArtLoadNow decodes every game's art for them, kept until the next scale)
void Load(IWICImagingFactory* wic, const wstring& job, bool trim = true) {
    const bool wantBig = job.rfind(L"big:", 0) == 0;
    const wstring appid = wantBig ? job.substr(4) : job;
    float scale = 1;
    {
        std::lock_guard<std::mutex> l(st.m);
        if (wantBig ? st.big.count(appid) > 0 : st.done.count(appid) > 0) { st.queued.erase(job); return; }
        scale = st.scale;
    }
    const wstring dir = st.cache.empty() ? wstring() : Join(st.cache, appid);
    const bool have = !dir.empty() && IsDir(dir);
    // whether the cache has the game's hero, answered here off the window's thread (HeroOnDisk reads it from paint)
    const wstring heroFile = have ? FindArt(dir, L"library_hero.jpg") : wstring();
    if (!dir.empty()) { std::lock_guard<std::mutex> l(st.m); st.heroOnDisk[appid] = !heroFile.empty(); }
    if (wantBig) {  // the featured banner's and the game page's hero, at the banner's width
        ImageP hero = Decode(wic, heroFile, std::min<UINT>(1920, Px(kBannerW, scale)));
        std::lock_guard<std::mutex> l(st.m);
        st.queued.erase(job);
        if (std::fabs(scale - st.scale) >= 0.01f) { st.oldBig[appid] = hero; Queue(job); return; }  // the scale changed meanwhile: again
        st.big[appid] = hero;
        st.oldBig.erase(appid);
        if (trim) Trim(st.big, st.bigUsed, kKeepBig);
        return;
    }
    Art a;
    if (have) {
        // the portrait capsule: 600 x 900 in the older cache layout, library_capsule in the newer one; 400 wide covers the
        // game page's 88 DIP cover at 300 %
        wstring coverFile = FindArt(dir, L"library_600x900.jpg");
        if (coverFile.empty()) coverFile = FindArt(dir, L"library_capsule.jpg");
        a.cover = Decode(wic, coverFile, 400);
        a.hero = Decode(wic, heroFile, Px(kTileArtW, scale));
        a.logo = Decode(wic, FindArt(dir, L"logo.png"), Px(kLogoW, scale));
        a.header = Decode(wic, FindArt(dir, L"header.jpg"), std::min<UINT>(460, Px(kTileArtW, scale)));
    }
    a.done = true;
    std::lock_guard<std::mutex> l(st.m);
    st.queued.erase(job);
    if (std::fabs(scale - st.scale) >= 0.01f) { st.oldDone[appid] = a; Queue(job); return; }  // the scale changed meanwhile: again
    st.done[appid] = a;
    st.oldDone.erase(appid);
    if (trim) Trim(st.done, st.used, kKeepArt);
}

void Worker() {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    IWICImagingFactory* wic = nullptr;
    CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic));
    for (;;) {
        wstring job;
        {
            std::unique_lock<std::mutex> l(st.m);
            st.cv.wait(l, [] { return st.stop || !st.queue.empty(); });
            if (st.stop) break;
            job = st.queue.front();
            st.queue.pop_front();
        }
        // a throw while decoding (out of memory on a huge or broken image) never ends the process (std::terminate from a
        // thread): the game keeps its generated plate, as for art that isn't there
        try { Load(wic, job); }
        catch (...) {
            const bool big = job.rfind(L"big:", 0) == 0;
            const wstring appid = big ? job.substr(4) : job;
            std::lock_guard<std::mutex> l(st.m);
            st.queued.erase(job);
            if (big) st.big[appid] = nullptr;
            else { Art none; none.done = true; st.done[appid] = none; }
        }
        HWND notify;
        UINT msg;
        {
            std::lock_guard<std::mutex> l(st.m);  // ArtStop clears it under the lock
            notify = st.notify; msg = st.msg;
        }
        if (notify) PostMessageW(notify, msg, 0, 0);
    }
    if (wic) wic->Release();
    CoUninitialize();
}

bool ValidAppId(const wstring& id) {
    if (id.empty() || id.size() > 12) return false;
    for (wchar_t c : id) if (c < L'0' || c > L'9') return false;
    return true;
}

}  // namespace

void ArtStart(HWND notify, UINT msg) {
    std::lock_guard<std::mutex> l(st.m);
    st.notify = notify; st.msg = msg;
    if (st.cache.empty()) st.cache = LibraryCache();
    if (st.running) return;
    st.running = true;
    st.stop = false;
    st.worker = std::thread(Worker);
}

void ArtStop() {
    {
        std::lock_guard<std::mutex> l(st.m);
        if (!st.running) return;
        st.stop = true;
        st.notify = nullptr;
    }
    st.cv.notify_all();
    if (st.worker.joinable()) st.worker.join();
    st.running = false;
}

// ui-render: the art a frame leaves out to show a fallback (View::artHide: 1 the hero and logo, 2 the header, 4 the cover)
static void Hide(const wstring& appid, Art& a) {
    if (!v.fixture) return;
    const auto it = v.artHide.find(appid);
    if (it == v.artHide.end()) return;
    if (it->second & 1) { a.hero = nullptr; a.logo = nullptr; }
    if (it->second & 2) a.header = nullptr;
    if (it->second & 4) a.cover = nullptr;
}

Art ArtFor(const wstring& appid) {
    if (!ValidAppId(appid)) { Art none; none.done = true; return none; }
    Art a;
    {
        std::lock_guard<std::mutex> l(st.m);
        auto it = st.done.find(appid);
        if (it != st.done.end()) a = it->second;
        st.used[appid] = ++st.tick;
        if (!a.done) {
            Queue(appid);
            const auto old = st.oldDone.find(appid);  // decoded at the scale before: drawn (scaled) until the new decode arrives
            if (old != st.oldDone.end()) a = old->second;
        }
    }
    Hide(appid, a);
    return a;
}

ImageP ArtBig(const wstring& appid, bool* done) {
    if (done) *done = true;
    if (!ValidAppId(appid)) return nullptr;
    ImageP img;
    {
        std::lock_guard<std::mutex> l(st.m);
        auto it = st.big.find(appid);
        st.bigUsed[appid] = ++st.tick;
        if (it != st.big.end()) img = it->second;
        else {
            Queue(L"big:" + appid);
            if (done) *done = false;
            const auto old = st.oldBig.find(appid);  // the scale before's, until the new one arrives
            if (old != st.oldBig.end()) img = old->second;
        }
    }
    if (v.fixture) { const auto h = v.artHide.find(appid); if (h != v.artHide.end() && (h->second & 1)) return nullptr; }
    return img;
}

// Whether the game has a hero in the Steam cache, before it is decoded: the game page keeps its banner layout while the
// art decodes. The art worker answers it as it looks in the game's cache folder (Load: the Library's lockups have asked
// already, as a rule); a DPI change keeps the answer. Not answered yet while the worker runs: the game's art is queued and
// the page waits on the banner's plate (a Steam game has its hero as a rule; the worker's message brings the answer), so
// paint never waits on a disk that has spun down. Without the worker (the offscreen renders): one look here.
bool HeroOnDisk(const wstring& appid) {
    if (!ValidAppId(appid)) return false;
    if (v.fixture) { const auto h = v.artHide.find(appid); if (h != v.artHide.end() && (h->second & 1)) return false; }
    wstring cache;
    {
        std::lock_guard<std::mutex> l(st.m);
        const auto it = st.heroOnDisk.find(appid);
        if (it != st.heroOnDisk.end()) return it->second;
        cache = st.cache;
        if (st.running && !cache.empty()) { Queue(appid); return true; }
    }
    if (cache.empty()) return false;  // no Steam (or not started yet): asked again next time
    const wstring dir = Join(cache, appid);
    const bool has = IsDir(dir) && !FindArt(dir, L"library_hero.jpg").empty();
    std::lock_guard<std::mutex> l(st.m);
    st.heroOnDisk[appid] = has;
    return has;
}

void ArtScale(float scale) {
    std::lock_guard<std::mutex> l(st.m);
    if (std::fabs(scale - st.scale) < 0.01f) return;
    st.scale = scale;
    // decoded again at the new size as they are drawn; until then the newest old decode of each is drawn (kept to about
    // twice what is kept decoded: a second change soon after the first keeps what had not come back yet)
    if (st.oldDone.size() + st.done.size() > 2 * kKeepArt) st.oldDone.clear();
    if (st.oldBig.size() + st.big.size() > 2 * kKeepBig) st.oldBig.clear();
    for (auto& d : st.done) st.oldDone[d.first] = d.second;
    for (auto& d : st.big) st.oldBig[d.first] = d.second;
    st.done.clear(); st.big.clear(); st.used.clear(); st.bigUsed.clear();
}

// A banner's hero (the featured banner, the game page's banner): the banner-sized decode, else the tile-sized one while it
// decodes. One that arrives after the banner was drawn without it fades in over what was there (kHeroFadeMs) instead of
// popping in: the sharp hero over the soft one, the first art over the banner's fallback (`under`). At once with Animation
// effects off and in still frames. Per game (UI thread, the few banners drawn last): the image last drawn, the one it fades
// from and when that began.
constexpr float kHeroFadeMs = 200;
struct HeroSeen { const Image* shown = nullptr; bool seen = false; ImageP from; bool fromUnder = false; ULONGLONG at = 0; unsigned long long used = 0; };
static std::map<wstring, HeroSeen> heroSeen;
static unsigned long long heroTick = 0;
static ULONGLONG heroFadeAt = 0;
bool HeroFading() { return heroFadeAt && (long long)NowMs() - (long long)heroFadeAt < (long long)kHeroFadeMs + 20; }
// A hero cover-fit at `focusY` as ImageCover draws it, but never below the top `keep` of the art's height: a Steam library
// hero ends in a dark ground band (Cyberpunk 2077's last 25 of 620 rows are black), which would read as a black strip, a
// letterbox, wherever a banner is wide enough to show the art's whole height (a narrow window). The art is scaled up a
// little more there instead; a wide banner crops as before. keep 1: ImageCover itself.
static void CoverKeep(Gfx& gx, const ImageP& img, const Rc& r, float rad, float focusY, float opacity, float keep) {
    if (keep >= 1) { gx.ImageCover(img, r, rad, focusY, opacity); return; }
    if (gx.dry || !img || img->w <= 0 || img->h <= 0) return;
    const float iw = (float)img->w, ih = (float)img->h;
    const float k = std::max(r.w / iw, r.h / (ih * keep));
    const float sh = r.h / k, sy = std::min((ih - sh) * focusY, ih * keep - sh);
    gx.PushRoundClip(r, rad);
    gx.ImageCover(img, Rc{r.x + (r.w - iw * k) / 2, r.y - sy * k, iw * k, ih * k}, 0, 0, opacity);  // the whole art at that scale, clipped
    gx.PopRoundClip();
}
ImageP HeroCover(Gfx& gx, const wstring& appid, const Rc& r, float rad, float focusY, const std::function<void()>& under, float* firstFade, float keep) {
    if (firstFade) *firstFade = 1;
    ImageP img = appid.empty() ? nullptr : ArtBig(appid);
    if (!img && !appid.empty()) img = ArtFor(appid).hero;
    float t = 1;
    ImageP from;
    bool fromUnder = false;
    if (v.live && !gx.dry && !appid.empty()) {
        if (heroSeen.size() > 4 && !heroSeen.count(appid)) {  // the banner drawn longest ago goes (it holds a decoded hero)
            auto oldest = heroSeen.begin();
            for (auto it = heroSeen.begin(); it != heroSeen.end(); ++it) if (it->second.used < oldest->second.used) oldest = it;
            heroSeen.erase(oldest);
        }
        HeroSeen& s = heroSeen[appid];
        s.used = ++heroTick;
        const ULONGLONG now = NowMs();
        if (img.get() != s.shown) {
            // a new image: it fades in over the one drawn before (or the fallback), unless this is the banner's first frame;
            // one more arriving while that fades carries the same fade on to it
            if (!s.at && s.seen && MotionOn() && v.motionAt < 0) {
                if (!(s.shown && s.from && s.from.get() == s.shown)) s.from = nullptr;
                s.fromUnder = !s.from;
                s.at = now;
            }
            s.shown = img.get();
        }
        s.seen = true;
        if (s.at) {
            t = (float)((long long)now - (long long)s.at) / kHeroFadeMs;
            if (t >= 1 || !img) { t = 1; s.at = 0; s.from = nullptr; }
            else { from = s.from; fromUnder = s.fromUnder; heroFadeAt = std::max(heroFadeAt, s.at); }
        }
        if (s.at == 0) s.from = img;  // what a later image fades from (held while it is drawn: the decoded art may go meanwhile)
        t = std::max(0.f, std::min(1.f, t));
    }
    if (!img) { if (under) under(); return nullptr; }
    if (t < 1) {
        if (from) CoverKeep(gx, from, r, rad, focusY, 1, keep);
        else if (fromUnder && under) under();
        if (fromUnder && firstFade) *firstFade = t;
    }
    CoverKeep(gx, img, r, rad, focusY, t, keep);
    return img;
}

void ArtLoadNow(const std::vector<wstring>& appids) {
    {
        std::lock_guard<std::mutex> l(st.m);
        if (st.cache.empty()) st.cache = LibraryCache();
    }
    IWICImagingFactory* wic = nullptr;
    CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic));
    // every game's art and banner-sized hero, none dropped (a frame's banner is never the tile's art stretched)
    for (auto& id : appids) if (ValidAppId(id)) { Load(wic, id, false); Load(wic, L"big:" + id, false); }
    std::lock_guard<std::mutex> l(st.m);
    st.oldDone.clear(); st.oldBig.clear();
    if (wic) wic->Release();
}

}  // namespace wg::app::lui
