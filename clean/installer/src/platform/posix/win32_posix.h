// SPDX-License-Identifier: MIT
// Linux: the Windows names wg_core and its tests use, declared the way Windows declares them, implemented on POSIX in
// win32_posix*.cpp. The core stays Windows-style source: a Windows call this layer lacks fails the Linux build at that line.
//
// Conventions (README-linux.md has the long form):
//  - Strings: wchar_t is 4 bytes here. Paths and names cross to the OS as UTF-8 (bytes that are not UTF-8 travel as U+DC80..U+DCFF
//    and come back unchanged); '\' becomes '/', a "\\?\" prefix goes, "Z:\" is "/", any other drive letter is ERROR_PATH_NOT_FOUND.
//  - REG_SZ sizes: callers write (length + 1) * 2 bytes (UTF-16); the store reads that as length + 1 wide characters.
//    On read the size is in sizeof(wchar_t) units, as the callers' own buffers are.
//  - Not available (fail cleanly): WinHTTP, COM objects, DXGI, the event log, LoadLibrary of any DLL, WinVerifyTrust.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <cwctype>
#include <ctime>
#include <strings.h>
#include <unistd.h>

// ---- types -------------------------------------------------------------------------------------------------------------
typedef uint32_t DWORD;
typedef int32_t LONG;
typedef uint32_t ULONG;
typedef int64_t LONGLONG;
typedef uint64_t ULONGLONG;
typedef uint64_t DWORD64;
typedef int BOOL;
typedef uint8_t BYTE;
typedef unsigned char BOOLEAN;
typedef uint16_t WORD;
typedef unsigned int UINT;
typedef uint32_t UINT32;
typedef uint64_t UINT64;
typedef intptr_t INT_PTR;
typedef uintptr_t ULONG_PTR;
typedef uintptr_t DWORD_PTR;
typedef size_t SIZE_T;
typedef int32_t HRESULT;
typedef LONG LSTATUS;
typedef LONG NTSTATUS;
typedef void* HANDLE;
typedef void* PVOID;
typedef void* LPVOID;
typedef const void* LPCVOID;
typedef HANDLE HMODULE;
typedef HANDLE HWND;
typedef HANDLE HINSTANCE;
typedef HANDLE HINTERNET;
typedef HANDLE HLOCAL;
typedef HANDLE EVT_HANDLE;
typedef BYTE* LPBYTE;
typedef DWORD* PDWORD;
typedef DWORD* LPDWORD;
typedef wchar_t WCHAR;
typedef wchar_t* LPWSTR;
typedef wchar_t* PWSTR;
typedef wchar_t* LPWCH;
typedef const wchar_t* LPCWSTR;
typedef const wchar_t* PCWSTR;
typedef const char* LPCSTR;
typedef char* LPSTR;
typedef INT_PTR (*FARPROC)();
typedef DWORD REGSAM;
typedef struct HKEY__* HKEY;
typedef HKEY* PHKEY;

#define WINAPI
#define CDECL
#define CALLBACK
#define APIENTRY
#define CONST const
#define TRUE 1
#define FALSE 0
#define MAX_PATH 260
#define MAKELONG(a, b) ((LONG)(((WORD)(a)) | ((DWORD)((WORD)(b))) << 16))
#define LOWORD(l) ((WORD)(((DWORD_PTR)(l)) & 0xffff))
#define HIWORD(l) ((WORD)((((DWORD_PTR)(l)) >> 16) & 0xffff))
#define INVALID_HANDLE_VALUE ((HANDLE)(intptr_t)-1)
#define SUCCEEDED(hr) (((HRESULT)(hr)) >= 0)
#define FAILED(hr) (((HRESULT)(hr)) < 0)
#define _countof(a) (sizeof(a) / sizeof((a)[0]))
#define ARRAYSIZE(a) _countof(a)
#define UNREFERENCED_PARAMETER(p) ((void)(p))

__extension__ typedef union _LARGE_INTEGER {
    struct { DWORD LowPart; LONG HighPart; };
    struct { DWORD LowPart; LONG HighPart; } u;
    LONGLONG QuadPart;
} LARGE_INTEGER;
__extension__ typedef union _ULARGE_INTEGER {
    struct { DWORD LowPart; DWORD HighPart; };
    struct { DWORD LowPart; DWORD HighPart; } u;
    ULONGLONG QuadPart;
} ULARGE_INTEGER;
typedef struct _FILETIME { DWORD dwLowDateTime, dwHighDateTime; } FILETIME;
typedef struct _SYSTEMTIME { WORD wYear, wMonth, wDayOfWeek, wDay, wHour, wMinute, wSecond, wMilliseconds; } SYSTEMTIME;
typedef struct _GUID { uint32_t Data1; uint16_t Data2, Data3; uint8_t Data4[8]; } GUID;
typedef GUID IID;
typedef GUID CLSID;
typedef GUID KNOWNFOLDERID;
typedef const GUID& REFIID;
typedef const GUID& REFCLSID;
typedef const GUID& REFKNOWNFOLDERID;
typedef struct _SECURITY_ATTRIBUTES { DWORD nLength; LPVOID lpSecurityDescriptor; BOOL bInheritHandle; } SECURITY_ATTRIBUTES, *LPSECURITY_ATTRIBUTES;
typedef PVOID PSECURITY_DESCRIPTOR;
typedef struct _OVERLAPPED OVERLAPPED;

// ---- errors, HRESULTs --------------------------------------------------------------------------------------------------
constexpr DWORD ERROR_SUCCESS = 0;
constexpr DWORD ERROR_INVALID_FUNCTION = 1;
constexpr DWORD ERROR_WRITE_PROTECT = 19;
constexpr DWORD ERROR_BAD_LENGTH = 24;
constexpr DWORD ERROR_READ_FAULT = 30;
constexpr DWORD ERROR_WRITE_FAULT = 29;
constexpr DWORD ERROR_GEN_FAILURE = 31;
constexpr DWORD ERROR_BAD_EXE_FORMAT = 193;
constexpr DWORD ERROR_ENVVAR_NOT_FOUND = 203;
constexpr DWORD ERROR_FILE_INVALID = 1006;
constexpr DWORD ERROR_NO_UNICODE_TRANSLATION = 1113;
constexpr DWORD ERROR_UNSUPPORTED_TYPE = 1630;
constexpr DWORD ERROR_RESOURCE_LANG_NOT_FOUND = 1815;
constexpr DWORD ERROR_RESOURCE_DATA_NOT_FOUND = 1812;
constexpr DWORD ERROR_FILE_NOT_FOUND = 2;
constexpr DWORD ERROR_PATH_NOT_FOUND = 3;
constexpr DWORD ERROR_TOO_MANY_OPEN_FILES = 4;
constexpr DWORD ERROR_ACCESS_DENIED = 5;
constexpr DWORD ERROR_INVALID_HANDLE = 6;
constexpr DWORD ERROR_NOT_ENOUGH_MEMORY = 8;
constexpr DWORD ERROR_INVALID_DATA = 13;
constexpr DWORD ERROR_OUTOFMEMORY = 14;
constexpr DWORD ERROR_NOT_SAME_DEVICE = 17;
constexpr DWORD ERROR_NO_MORE_FILES = 18;
constexpr DWORD ERROR_SHARING_VIOLATION = 32;
constexpr DWORD ERROR_LOCK_VIOLATION = 33;
constexpr DWORD ERROR_NOT_SUPPORTED = 50;
constexpr DWORD ERROR_FILE_EXISTS = 80;
constexpr DWORD ERROR_INVALID_PARAMETER = 87;
constexpr DWORD ERROR_BROKEN_PIPE = 109;
constexpr DWORD ERROR_DISK_FULL = 112;
constexpr DWORD ERROR_CALL_NOT_IMPLEMENTED = 120;
constexpr DWORD ERROR_INSUFFICIENT_BUFFER = 122;
constexpr DWORD ERROR_INVALID_NAME = 123;
constexpr DWORD ERROR_DIR_NOT_EMPTY = 145;
constexpr DWORD ERROR_ALREADY_EXISTS = 183;
constexpr DWORD ERROR_UNABLE_TO_REMOVE_REPLACED = 1175;
constexpr DWORD ERROR_UNABLE_TO_MOVE_REPLACEMENT = 1176;
constexpr DWORD ERROR_UNABLE_TO_MOVE_REPLACEMENT_2 = 1177;
constexpr DWORD ERROR_MORE_DATA = 234;
constexpr DWORD ERROR_NO_MORE_ITEMS = 259;
constexpr DWORD ERROR_DIRECTORY = 267;
constexpr DWORD ERROR_CANCELLED = 1223;
constexpr DWORD ERROR_TOO_MANY_LINKS = 1142;
constexpr DWORD ERROR_FILE_CORRUPT = 1392;
constexpr DWORD ERROR_PROC_NOT_FOUND = 127;
constexpr DWORD ERROR_MOD_NOT_FOUND = 126;
constexpr DWORD ERROR_NOT_FOUND = 1168;
constexpr DWORD ERROR_NO_ASSOCIATION = 1155;
constexpr DWORD ERROR_TIMEOUT = 1460;
constexpr DWORD ERROR_BAD_PATHNAME = 161;
constexpr DWORD ERROR_FILENAME_EXCED_RANGE = 206;
constexpr DWORD ERROR_FILE_TOO_LARGE = 223;
constexpr DWORD ERROR_NOT_ENOUGH_QUOTA = 1816;
constexpr DWORD ERROR_CANT_ACCESS_FILE = 1920;
constexpr DWORD ERROR_OPERATION_ABORTED = 995;
constexpr DWORD ERROR_IO_PENDING = 997;
constexpr DWORD ERROR_INTERNET_CANNOT_CONNECT = 12029;
constexpr DWORD ERROR_WINHTTP_CANNOT_CONNECT = 12029;
constexpr DWORD ERROR_WINHTTP_TIMEOUT = 12002;
constexpr DWORD ERROR_WINHTTP_INVALID_URL = 12005;
constexpr DWORD ERROR_WINHTTP_UNRECOGNIZED_SCHEME = 12006;
constexpr DWORD ERROR_WINHTTP_NAME_NOT_RESOLVED = 12007;
constexpr DWORD ERROR_WINHTTP_CONNECTION_ERROR = 12030;
constexpr DWORD ERROR_PRIVILEGE_NOT_HELD = 1314;
constexpr DWORD RPC_E_CHANGED_MODE = 0x80010106u;
constexpr DWORD ERROR_ELEVATION_REQUIRED = 740;
constexpr HRESULT S_OK = 0;
constexpr HRESULT S_FALSE = 1;
constexpr HRESULT E_FAIL = (HRESULT)0x80004005;
constexpr HRESULT E_NOTIMPL = (HRESULT)0x80004001;
constexpr HRESULT E_NOINTERFACE = (HRESULT)0x80004002;
constexpr HRESULT E_POINTER = (HRESULT)0x80004003;
constexpr HRESULT E_OUTOFMEMORY = (HRESULT)0x8007000E;
constexpr HRESULT E_INVALIDARG = (HRESULT)0x80070057;
constexpr HRESULT E_ACCESSDENIED = (HRESULT)0x80070005;
constexpr HRESULT REGDB_E_CLASSNOTREG = (HRESULT)0x80040154;
constexpr HRESULT DXGI_ERROR_NOT_FOUND = (HRESULT)0x887A0002;
constexpr HRESULT DXGI_ERROR_UNSUPPORTED = (HRESULT)0x887A0004;
constexpr HRESULT HRESULT_ERROR_FILE_NOT_FOUND = (HRESULT)0x80070002;
DWORD GetLastError();
void SetLastError(DWORD e);

// ---- files -------------------------------------------------------------------------------------------------------------
constexpr DWORD GENERIC_READ = 0x80000000u;
constexpr DWORD GENERIC_WRITE = 0x40000000u;
constexpr DWORD GENERIC_EXECUTE = 0x20000000u;
constexpr DWORD GENERIC_ALL = 0x10000000u;
constexpr DWORD DELETE = 0x00010000;
constexpr DWORD FILE_READ_DATA = 1;
constexpr DWORD FILE_WRITE_DATA = 2;
constexpr DWORD FILE_APPEND_DATA = 4;
constexpr DWORD FILE_READ_ATTRIBUTES = 0x80;
constexpr DWORD FILE_WRITE_ATTRIBUTES = 0x100;
constexpr DWORD SYNCHRONIZE = 0x00100000;
constexpr DWORD FILE_GENERIC_READ = 0x00120089;
constexpr DWORD FILE_GENERIC_WRITE = 0x00120116;
constexpr DWORD FILE_SHARE_READ = 1;
constexpr DWORD FILE_SHARE_WRITE = 2;
constexpr DWORD FILE_SHARE_DELETE = 4;
constexpr DWORD CREATE_NEW = 1;
constexpr DWORD CREATE_ALWAYS = 2;
constexpr DWORD OPEN_EXISTING = 3;
constexpr DWORD OPEN_ALWAYS = 4;
constexpr DWORD TRUNCATE_EXISTING = 5;
constexpr DWORD FILE_ATTRIBUTE_READONLY = 0x1;
constexpr DWORD FILE_ATTRIBUTE_HIDDEN = 0x2;
constexpr DWORD FILE_ATTRIBUTE_SYSTEM = 0x4;
constexpr DWORD FILE_ATTRIBUTE_DIRECTORY = 0x10;
constexpr DWORD FILE_ATTRIBUTE_ARCHIVE = 0x20;
constexpr DWORD FILE_ATTRIBUTE_NORMAL = 0x80;
constexpr DWORD FILE_ATTRIBUTE_TEMPORARY = 0x100;
constexpr DWORD FILE_ATTRIBUTE_REPARSE_POINT = 0x400;
constexpr DWORD FILE_FLAG_WRITE_THROUGH = 0x80000000u;
constexpr DWORD FILE_FLAG_OVERLAPPED = 0x40000000;
constexpr DWORD FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000;
constexpr DWORD FILE_FLAG_RANDOM_ACCESS = 0x10000000;
constexpr DWORD FILE_FLAG_DELETE_ON_CLOSE = 0x04000000;
constexpr DWORD FILE_FLAG_BACKUP_SEMANTICS = 0x02000000;
constexpr DWORD FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000;
constexpr DWORD FILE_BEGIN = 0;
constexpr DWORD FILE_CURRENT = 1;
constexpr DWORD FILE_END = 2;
constexpr DWORD INVALID_FILE_ATTRIBUTES = 0xFFFFFFFFu;
constexpr DWORD INVALID_SET_FILE_POINTER = 0xFFFFFFFFu;
constexpr DWORD MOVEFILE_REPLACE_EXISTING = 1;
constexpr DWORD MOVEFILE_COPY_ALLOWED = 2;
constexpr DWORD MOVEFILE_DELAY_UNTIL_REBOOT = 4;
constexpr DWORD MOVEFILE_WRITE_THROUGH = 8;
constexpr DWORD REPLACEFILE_WRITE_THROUGH = 1;
constexpr DWORD REPLACEFILE_IGNORE_MERGE_ERRORS = 2;
constexpr DWORD REPLACEFILE_IGNORE_ACL_ERRORS = 4;
constexpr DWORD PAGE_READONLY = 0x02;
constexpr DWORD PAGE_READWRITE = 0x04;
constexpr DWORD PAGE_WRITECOPY = 0x08;
constexpr DWORD FILE_MAP_COPY = 1;
constexpr DWORD FILE_MAP_WRITE = 2;
constexpr DWORD FILE_MAP_READ = 4;
constexpr DWORD SYMBOLIC_LINK_FLAG_DIRECTORY = 1;
constexpr DWORD SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE = 2;
constexpr DWORD FILE_NAME_NORMALIZED = 0;
constexpr DWORD FILE_NAME_OPENED = 8;
constexpr DWORD VOLUME_NAME_DOS = 0;
constexpr DWORD VOLUME_NAME_NONE = 4;
constexpr DWORD FIND_FIRST_EX_LARGE_FETCH = 2;
typedef enum _GET_FILEEX_INFO_LEVELS { GetFileExInfoStandard = 0 } GET_FILEEX_INFO_LEVELS;
typedef enum _FINDEX_INFO_LEVELS { FindExInfoStandard = 0, FindExInfoBasic = 1 } FINDEX_INFO_LEVELS;
typedef enum _FINDEX_SEARCH_OPS { FindExSearchNameMatch = 0 } FINDEX_SEARCH_OPS;
typedef enum _FILE_INFO_BY_HANDLE_CLASS { FileBasicInfo = 0, FileRenameInfo = 3, FileDispositionInfo = 4, FileIdInfo = 18, FileDispositionInfoEx = 21 } FILE_INFO_BY_HANDLE_CLASS;
typedef struct _FILE_BASIC_INFO {
    LARGE_INTEGER CreationTime, LastAccessTime, LastWriteTime, ChangeTime;
    DWORD FileAttributes;
} FILE_BASIC_INFO;
typedef struct _FILE_DISPOSITION_INFO { BOOLEAN DeleteFile; } FILE_DISPOSITION_INFO;
// FileRenameInfo: the file of the handle (opened with DELETE access) takes the name FileName (FileNameLength bytes, no NUL; a
// full path, also in the NT form \??\Z:\...). Linux: rename(2) of the name /proc/self/fd gives for the handle, so the file
// renamed is the handle's own unless another process renames it in that instant; ReplaceIfExists off is renameat2's
// RENAME_NOREPLACE (a file system without it: link and unlink, else a check before the rename). RootDirectory must be null.
typedef struct _FILE_RENAME_INFO {
    union { BOOLEAN ReplaceIfExists; DWORD Flags; };
    HANDLE RootDirectory;
    DWORD FileNameLength;
    WCHAR FileName[1];
} FILE_RENAME_INFO;
typedef struct _FILE_DISPOSITION_INFO_EX { DWORD Flags; } FILE_DISPOSITION_INFO_EX;
constexpr DWORD FILE_DISPOSITION_FLAG_DO_NOT_DELETE = 0x0;
constexpr DWORD FILE_DISPOSITION_FLAG_DELETE = 0x1;
constexpr DWORD FILE_DISPOSITION_FLAG_POSIX_SEMANTICS = 0x2;
constexpr DWORD FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE = 0x10;
typedef struct _WIN32_FILE_ATTRIBUTE_DATA {
    DWORD dwFileAttributes; FILETIME ftCreationTime, ftLastAccessTime, ftLastWriteTime; DWORD nFileSizeHigh, nFileSizeLow;
} WIN32_FILE_ATTRIBUTE_DATA;
typedef struct _WIN32_FIND_DATAW {
    DWORD dwFileAttributes; FILETIME ftCreationTime, ftLastAccessTime, ftLastWriteTime; DWORD nFileSizeHigh, nFileSizeLow;
    DWORD dwReserved0, dwReserved1; wchar_t cFileName[MAX_PATH]; wchar_t cAlternateFileName[14];
} WIN32_FIND_DATAW;
typedef struct _BY_HANDLE_FILE_INFORMATION {
    DWORD dwFileAttributes; FILETIME ftCreationTime, ftLastAccessTime, ftLastWriteTime;
    DWORD dwVolumeSerialNumber, nFileSizeHigh, nFileSizeLow, nNumberOfLinks, nFileIndexHigh, nFileIndexLow;
} BY_HANDLE_FILE_INFORMATION;
typedef struct _FILE_ID_128 { BYTE Identifier[16]; } FILE_ID_128;
typedef struct _FILE_ID_INFO { ULONGLONG VolumeSerialNumber; FILE_ID_128 FileId; } FILE_ID_INFO;

// Sharing rules between handles of this process as Windows has them. Between processes only Windows' image rule: an open that
// would write (write or append data, or a truncating disposition) to a Windows program or DLL ("MZ") that a Windows program
// under Wine has loaded fails with ERROR_SHARING_VIOLATION, as Windows refuses a write open of a loaded image (what the
// operation guard relies on to see a running game). Looked at: this user's Wine processes (command line a Windows path, an
// .exe, or wine*), their /proc/<pid>/maps. One of them whose maps cannot be read: ERROR_CANT_ACCESS_FILE ("could not tell").
// Not seen: another user's processes; a file another program merely has open (its share mode is not known here); a game
// that loads the file after the open succeeded (Windows refuses that load while the write handle is open; Linux cannot).
HANDLE CreateFileW(LPCWSTR name, DWORD access, DWORD share, LPSECURITY_ATTRIBUTES sa, DWORD disposition, DWORD flags, HANDLE templ);
BOOL ReadFile(HANDLE h, LPVOID buf, DWORD n, LPDWORD got, OVERLAPPED* ov);
BOOL WriteFile(HANDLE h, LPCVOID buf, DWORD n, LPDWORD wrote, OVERLAPPED* ov);
BOOL CloseHandle(HANDLE h);
BOOL FlushFileBuffers(HANDLE h);
BOOL GetFileSizeEx(HANDLE h, LARGE_INTEGER* size);
BOOL SetFilePointerEx(HANDLE h, LARGE_INTEGER dist, LARGE_INTEGER* newPos, DWORD method);
BOOL GetFileTime(HANDLE h, FILETIME* created, FILETIME* accessed, FILETIME* written);
BOOL SetFileTime(HANDLE h, const FILETIME* created, const FILETIME* accessed, const FILETIME* written);
BOOL GetFileInformationByHandle(HANDLE h, BY_HANDLE_FILE_INFORMATION* info);
BOOL GetFileInformationByHandleEx(HANDLE h, FILE_INFO_BY_HANDLE_CLASS cls, LPVOID info, DWORD size);
BOOL SetFileInformationByHandle(HANDLE h, FILE_INFO_BY_HANDLE_CLASS cls, LPVOID info, DWORD size);
BOOL SetEndOfFile(HANDLE h);
DWORD GetFinalPathNameByHandleW(HANDLE h, LPWSTR out, DWORD size, DWORD flags);
DWORD GetFileAttributesW(LPCWSTR name);
BOOL GetFileAttributesExW(LPCWSTR name, GET_FILEEX_INFO_LEVELS level, LPVOID data);
BOOL SetFileAttributesW(LPCWSTR name, DWORD attrs);
BOOL DeleteFileW(LPCWSTR name);
BOOL MoveFileExW(LPCWSTR from, LPCWSTR to, DWORD flags);
BOOL ReplaceFileW(LPCWSTR replaced, LPCWSTR replacement, LPCWSTR backup, DWORD flags, LPVOID exclude, LPVOID reserved);
BOOL CopyFileW(LPCWSTR from, LPCWSTR to, BOOL failIfExists);
BOOL CreateDirectoryW(LPCWSTR name, LPSECURITY_ATTRIBUTES sa);
BOOL RemoveDirectoryW(LPCWSTR name);
BOOL CreateHardLinkW(LPCWSTR link, LPCWSTR existing, LPSECURITY_ATTRIBUTES sa);
HANDLE FindFirstFileW(LPCWSTR pattern, WIN32_FIND_DATAW* data);
HANDLE FindFirstFileExW(LPCWSTR pattern, FINDEX_INFO_LEVELS info, LPVOID data, FINDEX_SEARCH_OPS op, LPVOID filter, DWORD flags);
BOOL FindNextFileW(HANDLE h, WIN32_FIND_DATAW* data);
BOOL FindClose(HANDLE h);
DWORD GetFullPathNameW(LPCWSTR name, DWORD size, LPWSTR out, LPWSTR* filePart);
// no 8.3 names here: the path as it is, its names in their case on disk (no link followed); fails as Windows does when it is not there
DWORD GetLongPathNameW(LPCWSTR shortPath, LPWSTR out, DWORD size);
DWORD GetTempPathW(DWORD size, LPWSTR out);
DWORD GetCurrentDirectoryW(DWORD size, LPWSTR out);
BOOL SetCurrentDirectoryW(LPCWSTR dir);
DWORD GetSystemDirectoryW(LPWSTR out, DWORD size);
DWORD GetModuleFileNameW(HMODULE m, LPWSTR out, DWORD size);
BOOL GetDiskFreeSpaceExW(LPCWSTR dir, ULARGE_INTEGER* available, ULARGE_INTEGER* total, ULARGE_INTEGER* freeBytes);
HANDLE CreateFileMappingW(HANDLE file, LPSECURITY_ATTRIBUTES sa, DWORD protect, DWORD sizeHigh, DWORD sizeLow, LPCWSTR name);
LPVOID MapViewOfFile(HANDLE map, DWORD access, DWORD offsetHigh, DWORD offsetLow, SIZE_T bytes);
BOOL UnmapViewOfFile(LPCVOID base);
BOOL CreateSymbolicLinkW(LPCWSTR link, LPCWSTR target, DWORD flags);
int _wsystem(const wchar_t* cmd);
constexpr DWORD COPY_FILE_FAIL_IF_EXISTS = 1;
constexpr DWORD COPY_FILE_COPY_SYMLINK = 0x800;
constexpr DWORD FILE_TYPE_UNKNOWN = 0;
constexpr DWORD FILE_TYPE_DISK = 1;
constexpr DWORD FILE_TYPE_CHAR = 2;
constexpr DWORD FILE_TYPE_PIPE = 3;
constexpr DWORD ERROR_MODE_SEM_FAILCRITICALERRORS = 1;
constexpr UINT SEM_FAILCRITICALERRORS = 1;
constexpr UINT SEM_NOOPENFILEERRORBOX = 0x8000;
BOOL CopyFileExW(LPCWSTR from, LPCWSTR to, void* progress, LPVOID data, BOOL* cancel, DWORD flags);
DWORD GetFileType(HANDLE h);
BOOL SetThreadErrorMode(DWORD mode, LPDWORD old);
#define STD_INPUT_HANDLE ((DWORD)-10)
#define STD_OUTPUT_HANDLE ((DWORD)-11)
#define STD_ERROR_HANDLE ((DWORD)-12)
HANDLE GetStdHandle(DWORD which);
DWORD GetLogicalDrives();   // only Z: exists, and it is /
FILE* _wfopen(const wchar_t* path, const wchar_t* mode);

// ---- strings -----------------------------------------------------------------------------------------------------------
constexpr UINT CP_ACP = 0;
constexpr UINT CP_UTF8 = 65001;
constexpr DWORD MB_ERR_INVALID_CHARS = 8;
constexpr DWORD WC_ERR_INVALID_CHARS = 0x80;
int MultiByteToWideChar(UINT cp, DWORD flags, const char* s, int n, wchar_t* out, int outN);
int WideCharToMultiByte(UINT cp, DWORD flags, const wchar_t* s, int n, char* out, int outN, const char* def, BOOL* used);
int _wcsicmp(const wchar_t* a, const wchar_t* b);
int _wcsnicmp(const wchar_t* a, const wchar_t* b, size_t n);
int _stricmp(const char* a, const char* b);
constexpr int CSTR_LESS_THAN = 1;
constexpr int CSTR_EQUAL = 2;
constexpr int CSTR_GREATER_THAN = 3;
int CompareStringOrdinal(const wchar_t* a, int an, const wchar_t* b, int bn, BOOL ignoreCase);
int sscanf_s(const char* s, const char* fmt, ...);   // the size argument after each %s, %c and %[ is read and dropped
int _wtoi(const wchar_t* s);
long long _wtoi64(const wchar_t* s);
#define wcsicmp _wcsicmp
#define stricmp _stricmp
constexpr DWORD FORMAT_MESSAGE_ALLOCATE_BUFFER = 0x100;
constexpr DWORD FORMAT_MESSAGE_IGNORE_INSERTS = 0x200;
constexpr DWORD FORMAT_MESSAGE_FROM_SYSTEM = 0x1000;
DWORD FormatMessageW(DWORD flags, LPCVOID src, DWORD id, DWORD lang, LPWSTR buf, DWORD size, void* args);
HLOCAL LocalFree(HLOCAL p);
// Security descriptors do not exist here: always FALSE (ERROR_NOT_SUPPORTED), *sd null. Callers that go on without one get
// the POSIX permissions instead (a named mutex: its lock file is the user's own, mode 0600).
constexpr DWORD SDDL_REVISION_1 = 1;
BOOL ConvertStringSecurityDescriptorToSecurityDescriptorW(LPCWSTR sddl, DWORD revision, PSECURITY_DESCRIPTOR* sd, ULONG* size);
HANDLE GetProcessHeap();
BOOL HeapFree(HANDLE heap, DWORD flags, LPVOID p);
[[noreturn]] void ExitProcess(UINT code);
DWORD GetEnvironmentVariableW(LPCWSTR name, LPWSTR out, DWORD size);
BOOL SetEnvironmentVariableW(LPCWSTR name, LPCWSTR value);
LPWCH GetEnvironmentStringsW();
BOOL FreeEnvironmentStringsW(LPWCH block);

// ---- time, sleep, ids ----------------------------------------------------------------------------------------------------
ULONGLONG GetTickCount64();
void GetLocalTime(SYSTEMTIME* t);
BOOL FileTimeToSystemTime(const FILETIME* ft, SYSTEMTIME* st);
BOOL SystemTimeToFileTime(const SYSTEMTIME* st, FILETIME* ft);
BOOL FileTimeToLocalFileTime(const FILETIME* ft, FILETIME* local);
BOOL LocalFileTimeToFileTime(const FILETIME* local, FILETIME* ft);
LONG CompareFileTime(const FILETIME* a, const FILETIME* b);
void Sleep(DWORD ms);
DWORD GetCurrentProcessId();
BOOL QueryPerformanceCounter(LARGE_INTEGER* v);
BOOL QueryPerformanceFrequency(LARGE_INTEGER* v);

// ---- processes, pipes, jobs ----------------------------------------------------------------------------------------------
constexpr DWORD CREATE_SUSPENDED = 0x4;
constexpr DWORD CREATE_NEW_CONSOLE = 0x10;
constexpr DWORD CREATE_NEW_PROCESS_GROUP = 0x200;
constexpr DWORD CREATE_UNICODE_ENVIRONMENT = 0x400;
constexpr DWORD CREATE_DEFAULT_ERROR_MODE = 0x04000000;
constexpr DWORD CREATE_NO_WINDOW = 0x08000000;
constexpr DWORD DETACHED_PROCESS = 0x8;
constexpr DWORD CREATE_BREAKAWAY_FROM_JOB = 0x01000000;
constexpr DWORD BELOW_NORMAL_PRIORITY_CLASS = 0x4000;
constexpr DWORD IDLE_PRIORITY_CLASS = 0x40;
constexpr DWORD NORMAL_PRIORITY_CLASS = 0x20;
constexpr DWORD ABOVE_NORMAL_PRIORITY_CLASS = 0x8000;
constexpr DWORD STARTF_USESHOWWINDOW = 1;
constexpr DWORD STARTF_USESTDHANDLES = 0x100;
constexpr DWORD HANDLE_FLAG_INHERIT = 1;
constexpr DWORD INFINITE = 0xFFFFFFFFu;
constexpr DWORD WAIT_OBJECT_0 = 0;
constexpr DWORD WAIT_TIMEOUT = 258;
constexpr DWORD WAIT_FAILED = 0xFFFFFFFFu;
constexpr DWORD WAIT_ABANDONED = 0x80;
constexpr DWORD STILL_ACTIVE = 259;
constexpr DWORD PROCESS_TERMINATE = 1;
constexpr DWORD PROCESS_QUERY_LIMITED_INFORMATION = 0x1000;
constexpr DWORD PROCESS_QUERY_INFORMATION = 0x400;
constexpr DWORD SYNCHRONIZE_ACCESS_UNUSED = 0;
constexpr DWORD TH32CS_SNAPPROCESS = 2;
constexpr DWORD JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000;
constexpr DWORD TOKEN_QUERY = 8;
constexpr DWORD SW_HIDE = 0;
constexpr DWORD SW_SHOWNORMAL = 1;
constexpr DWORD SW_SHOW = 5;
constexpr DWORD SW_SHOWNOACTIVATE = 4;
constexpr DWORD SW_SHOWMINNOACTIVE = 7;
constexpr DWORD SEE_MASK_NOCLOSEPROCESS = 0x40;
constexpr DWORD SEE_MASK_FLAG_NO_UI = 0x400;
constexpr DWORD SEE_MASK_NOASYNC = 0x100;
constexpr DWORD SEE_MASK_FLAG_DDEWAIT = 0x100;
constexpr DWORD COINIT_APARTMENTTHREADED = 2;
constexpr DWORD COINIT_MULTITHREADED = 0;
constexpr DWORD COINIT_DISABLE_OLE1DDE = 4;
constexpr DWORD CLSCTX_INPROC_SERVER = 1;
typedef struct _STARTUPINFOW {
    DWORD cb; LPWSTR lpReserved, lpDesktop, lpTitle; DWORD dwX, dwY, dwXSize, dwYSize, dwXCountChars, dwYCountChars, dwFillAttribute, dwFlags;
    WORD wShowWindow, cbReserved2; LPBYTE lpReserved2; HANDLE hStdInput, hStdOutput, hStdError;
} STARTUPINFOW;
typedef struct _PROCESS_INFORMATION { HANDLE hProcess, hThread; DWORD dwProcessId, dwThreadId; } PROCESS_INFORMATION;
typedef struct tagPROCESSENTRY32W {
    DWORD dwSize, cntUsage, th32ProcessID; ULONG_PTR th32DefaultHeapID; DWORD th32ModuleID, cntThreads, th32ParentProcessID;
    LONG pcPriClassBase; DWORD dwFlags; wchar_t szExeFile[MAX_PATH];
} PROCESSENTRY32W;
typedef enum _JOBOBJECTINFOCLASS { JobObjectBasicAccountingInformation = 1, JobObjectExtendedLimitInformation = 9 } JOBOBJECTINFOCLASS;
typedef struct _JOBOBJECT_BASIC_LIMIT_INFORMATION {
    LARGE_INTEGER PerProcessUserTimeLimit, PerJobUserTimeLimit; DWORD LimitFlags; SIZE_T MinimumWorkingSetSize, MaximumWorkingSetSize;
    DWORD ActiveProcessLimit; ULONG_PTR Affinity; DWORD PriorityClass, SchedulingClass;
} JOBOBJECT_BASIC_LIMIT_INFORMATION;
typedef struct _IO_COUNTERS { ULONGLONG ReadOperationCount, WriteOperationCount, OtherOperationCount, ReadTransferCount, WriteTransferCount, OtherTransferCount; } IO_COUNTERS;
typedef struct _JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
    JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation; IO_COUNTERS IoInfo;
    SIZE_T ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
} JOBOBJECT_EXTENDED_LIMIT_INFORMATION;
typedef struct _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION {
    LARGE_INTEGER TotalUserTime, TotalKernelTime, ThisPeriodTotalUserTime, ThisPeriodTotalKernelTime;
    DWORD TotalPageFaultCount, TotalProcesses, ActiveProcesses, TotalTerminatedProcesses;
} JOBOBJECT_BASIC_ACCOUNTING_INFORMATION;

BOOL CreateProcessW(LPCWSTR app, LPWSTR cmdline, LPSECURITY_ATTRIBUTES pa, LPSECURITY_ATTRIBUTES ta, BOOL inherit, DWORD flags,
                    LPVOID env, LPCWSTR dir, STARTUPINFOW* si, PROCESS_INFORMATION* pi);
DWORD WaitForSingleObject(HANDLE h, DWORD ms);   // a process, a mutex or an event

// ---- synchronization -----------------------------------------------------------------------------------------------------
// Named mutexes (win32_posix_sync.cpp): a lock file per name in this user's runtime folder ($XDG_RUNTIME_DIR, else the
// XDG data folder), with Linux open file description locks (fcntl F_OFD_SETLK) on two bytes: byte 0 read-locked by every
// handle (the object exists while one is open: ERROR_ALREADY_EXISTS, OpenMutexW), byte 1 write-locked by the owner. The kernel
// drops both when a process ends, so a dead owner never keeps it: the next owner gets WAIT_ABANDONED (the file's first bytes
// say "owned" until a ReleaseMutex). Owned by a thread and recursive for it, as on Windows; WaitForSingleObject's timeout is
// kept by polling (1 to 20 ms). Global\ and Local\ are the same per-user folder: two Linux accounts do not see each other's
// names, and a Windows program under Wine has its own (wineserver's) names. Unnamed mutexes are not supported. A thread that
// ends while it owns one keeps it until its process ends or closes every handle of it (Windows abandons it at the thread's end).
constexpr DWORD MUTEX_MODIFY_STATE = 0x0001;
constexpr DWORD ERROR_NOT_OWNER = 288;
HANDLE CreateMutexW(LPSECURITY_ATTRIBUTES sa, BOOL initialOwner, LPCWSTR name);
HANDLE OpenMutexW(DWORD access, BOOL inherit, LPCWSTR name);   // ERROR_FILE_NOT_FOUND when no handle of that name is open
BOOL ReleaseMutex(HANDLE h);                                  // ERROR_NOT_OWNER from a thread that does not own it
// Events: unnamed only, within this process (a name: ERROR_NOT_SUPPORTED).
HANDLE CreateEventW(LPSECURITY_ATTRIBUTES sa, BOOL manualReset, BOOL initialState, LPCWSTR name);
BOOL SetEvent(HANDLE h);
inline LONG InterlockedIncrement(volatile LONG* p) { return __atomic_add_fetch(p, 1, __ATOMIC_SEQ_CST); }
BOOL GetExitCodeProcess(HANDLE h, LPDWORD code);
BOOL TerminateProcess(HANDLE h, UINT code);
DWORD ResumeThread(HANDLE h);
HANDLE GetCurrentProcess();
BOOL CreatePipe(HANDLE* read, HANDLE* write, LPSECURITY_ATTRIBUTES sa, DWORD size);
BOOL PeekNamedPipe(HANDLE h, LPVOID buf, DWORD size, LPDWORD read, LPDWORD available, LPDWORD left);
BOOL SetHandleInformation(HANDLE h, DWORD mask, DWORD flags);
HANDLE CreateJobObjectW(LPSECURITY_ATTRIBUTES sa, LPCWSTR name);
BOOL SetInformationJobObject(HANDLE job, JOBOBJECTINFOCLASS cls, LPVOID info, DWORD size);
BOOL QueryInformationJobObject(HANDLE job, JOBOBJECTINFOCLASS cls, LPVOID info, DWORD size, LPDWORD returned);
BOOL AssignProcessToJobObject(HANDLE job, HANDLE process);
BOOL TerminateJobObject(HANDLE job, UINT code);
HANDLE CreateToolhelp32Snapshot(DWORD flags, DWORD pid);
BOOL Process32FirstW(HANDLE snap, PROCESSENTRY32W* pe);
BOOL Process32NextW(HANDLE snap, PROCESSENTRY32W* pe);
BOOL OpenProcessToken(HANDLE process, DWORD access, HANDLE* token);
HANDLE OpenProcess(DWORD access, BOOL inherit, DWORD pid);

// tokens and SIDs: this process's user is real (uid), "elevated" means root
typedef enum _TOKEN_INFORMATION_CLASS { TokenUser = 1, TokenElevation = 20 } TOKEN_INFORMATION_CLASS;
typedef struct _SID_PLACEHOLDER { uint32_t uid; } SID, *PSID;
typedef struct _SID_AND_ATTRIBUTES { PSID Sid; DWORD Attributes; } SID_AND_ATTRIBUTES;
typedef struct _TOKEN_USER { SID_AND_ATTRIBUTES User; } TOKEN_USER;
typedef struct _TOKEN_ELEVATION { DWORD TokenIsElevated; } TOKEN_ELEVATION;
BOOL GetTokenInformation(HANDLE token, TOKEN_INFORMATION_CLASS cls, LPVOID info, DWORD size, LPDWORD needed);
BOOL ConvertSidToStringSidW(PSID sid, LPWSTR* out);

// ---- modules (no Windows DLL exists here) -------------------------------------------------------------------------------
constexpr DWORD LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x800;
constexpr DWORD LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x1000;
constexpr DWORD LOAD_LIBRARY_SEARCH_APPLICATION_DIR = 0x200;
constexpr DWORD LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x100;
constexpr DWORD LOAD_WITH_ALTERED_SEARCH_PATH = 8;
HMODULE LoadLibraryExW(LPCWSTR name, HANDLE file, DWORD flags);
HMODULE GetModuleHandleW(LPCWSTR name);
FARPROC GetProcAddress(HMODULE m, LPCSTR name);

// ---- registry ----------------------------------------------------------------------------------------------------------
#define HKEY_CLASSES_ROOT ((HKEY)(ULONG_PTR)(LONG)0x80000000)
#define HKEY_CURRENT_USER ((HKEY)(ULONG_PTR)(LONG)0x80000001)
#define HKEY_LOCAL_MACHINE ((HKEY)(ULONG_PTR)(LONG)0x80000002)
#define HKEY_USERS ((HKEY)(ULONG_PTR)(LONG)0x80000003)
constexpr DWORD REG_NONE = 0;
constexpr DWORD REG_SZ = 1;
constexpr DWORD REG_EXPAND_SZ = 2;
constexpr DWORD REG_BINARY = 3;
constexpr DWORD REG_DWORD = 4;
constexpr DWORD REG_MULTI_SZ = 7;
constexpr DWORD REG_QWORD = 11;
constexpr DWORD KEY_QUERY_VALUE = 1;
constexpr DWORD KEY_SET_VALUE = 2;
constexpr DWORD KEY_CREATE_SUB_KEY = 4;
constexpr DWORD KEY_ENUMERATE_SUB_KEYS = 8;
constexpr DWORD KEY_NOTIFY = 0x10;
constexpr DWORD KEY_READ = 0x20019;
constexpr DWORD KEY_WRITE = 0x20006;
constexpr DWORD KEY_ALL_ACCESS = 0xF003F;
constexpr DWORD KEY_WOW64_32KEY = 0x200;
constexpr DWORD KEY_WOW64_64KEY = 0x100;
constexpr DWORD REG_OPTION_NON_VOLATILE = 0;
constexpr DWORD REG_OPTION_VOLATILE = 1;
constexpr DWORD REG_CREATED_NEW_KEY = 1;
constexpr DWORD REG_OPENED_EXISTING_KEY = 2;
constexpr DWORD RRF_RT_REG_SZ = 2;
constexpr DWORD RRF_RT_REG_DWORD = 0x10;
constexpr DWORD RRF_RT_ANY = 0xFFFF;
LSTATUS RegOpenKeyExW(HKEY key, LPCWSTR sub, DWORD options, REGSAM sam, PHKEY out);
LSTATUS RegCreateKeyExW(HKEY key, LPCWSTR sub, DWORD reserved, LPWSTR cls, DWORD options, REGSAM sam, LPSECURITY_ATTRIBUTES sa, PHKEY out, LPDWORD disposition);
LSTATUS RegCloseKey(HKEY key);
LSTATUS RegQueryValueExW(HKEY key, LPCWSTR name, LPDWORD reserved, LPDWORD type, LPBYTE data, LPDWORD size);
LSTATUS RegGetValueW(HKEY key, LPCWSTR sub, LPCWSTR name, DWORD flags, LPDWORD type, PVOID data, LPDWORD size);
LSTATUS RegSetValueExW(HKEY key, LPCWSTR name, DWORD reserved, DWORD type, const BYTE* data, DWORD size);
LSTATUS RegDeleteKeyW(HKEY key, LPCWSTR sub);
LSTATUS RegEnumKeyExW(HKEY key, DWORD index, LPWSTR name, LPDWORD nameLen, LPDWORD reserved, LPWSTR cls, LPDWORD classLen, FILETIME* written);
LSTATUS RegLoadAppKeyW(LPCWSTR file, PHKEY out, REGSAM sam, DWORD options, DWORD reserved);

// ---- shell -------------------------------------------------------------------------------------------------------------
extern const GUID FOLDERID_LocalAppData, FOLDERID_Desktop, FOLDERID_Documents, FOLDERID_Programs, FOLDERID_RoamingAppData,
    FOLDERID_ProgramData, FOLDERID_Profile, FOLDERID_StartMenu, FOLDERID_ProgramFiles, FOLDERID_ProgramFilesX86, FOLDERID_Downloads;
constexpr DWORD KF_FLAG_DEFAULT = 0;
constexpr DWORD KF_FLAG_CREATE = 0x8000;
constexpr DWORD KF_FLAG_DONT_VERIFY = 0x4000;
HRESULT SHGetKnownFolderPath(REFKNOWNFOLDERID id, DWORD flags, HANDLE token, PWSTR* out);
void CoTaskMemFree(LPVOID p);
LPVOID CoTaskMemAlloc(SIZE_T n);
HRESULT CoInitializeEx(LPVOID reserved, DWORD model);
void CoUninitialize();
typedef struct _SHELLEXECUTEINFOW {
    DWORD cbSize, fMask; HWND hwnd; LPCWSTR lpVerb, lpFile, lpParameters, lpDirectory; int nShow; HINSTANCE hInstApp;
    LPVOID lpIDList; LPCWSTR lpClass; HKEY hkeyClass; DWORD dwHotKey; HANDLE hIcon; HANDLE hProcess;
} SHELLEXECUTEINFOW;
BOOL ShellExecuteExW(SHELLEXECUTEINFOW* sei);

// COM interfaces the core names. Objects cannot be created (CoCreateInstance fails), so they are declarations only.
struct IUnknown {
    virtual HRESULT QueryInterface(REFIID id, void** out) = 0;
    virtual ULONG AddRef() = 0;
    virtual ULONG Release() = 0;
};
struct IPersistFile : IUnknown { virtual HRESULT Save(LPCWSTR file, BOOL remember) = 0; };
struct IShellLinkW : IUnknown {
    virtual HRESULT SetPath(LPCWSTR path) = 0;
    virtual HRESULT SetWorkingDirectory(LPCWSTR dir) = 0;
    virtual HRESULT SetDescription(LPCWSTR text) = 0;
    virtual HRESULT SetIconLocation(LPCWSTR path, int index) = 0;
    virtual HRESULT SetArguments(LPCWSTR args) = 0;
};
extern const GUID CLSID_ShellLink, IID_IShellLinkW, IID_IPersistFile;
HRESULT CoCreateInstance(REFCLSID cls, IUnknown* outer, DWORD context, REFIID iid, LPVOID* out);

// DXGI: interfaces only; CreateDXGIFactory1 is reached through GetProcAddress of a DLL that does not load here
struct IDXGIObject : IUnknown {};
struct IDXGIDevice : IDXGIObject {};
constexpr UINT DXGI_ADAPTER_FLAG_NONE = 0;
constexpr UINT DXGI_ADAPTER_FLAG_REMOTE = 1;
constexpr UINT DXGI_ADAPTER_FLAG_SOFTWARE = 2;
typedef struct _LUID { DWORD LowPart; LONG HighPart; } LUID;
typedef struct DXGI_ADAPTER_DESC1 {
    wchar_t Description[128]; UINT VendorId, DeviceId, SubSysId, Revision; SIZE_T DedicatedVideoMemory, DedicatedSystemMemory, SharedSystemMemory;
    LUID AdapterLuid; UINT Flags;
} DXGI_ADAPTER_DESC1;
struct IDXGIAdapter1 : IDXGIObject {
    virtual HRESULT CheckInterfaceSupport(REFIID id, LARGE_INTEGER* umd) = 0;
    virtual HRESULT GetDesc1(DXGI_ADAPTER_DESC1* desc) = 0;
};
struct IDXGIFactory1 : IDXGIObject { virtual HRESULT EnumAdapters1(UINT index, IDXGIAdapter1** out) = 0; };
template <class T> const GUID& wg_posix_uuidof();
#define __uuidof(T) (::wg_posix_uuidof<T>())

// event log (wevtapi): types for the function pointers the core casts GetProcAddress results to
constexpr DWORD EvtQueryChannelPath = 1;
constexpr DWORD EvtQueryReverseDirection = 0x200;
constexpr DWORD EvtRenderEventXml = 1;

// ---- WinHTTP: no network access in this build; every call that would start one fails ----------------------------------
constexpr DWORD WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY = 4;
constexpr DWORD WINHTTP_FLAG_SECURE = 0x00800000;
constexpr DWORD WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2 = 0x00000800;
constexpr DWORD WINHTTP_OPTION_SECURE_PROTOCOLS = 84;
constexpr DWORD WINHTTP_QUERY_STATUS_CODE = 19;
constexpr DWORD WINHTTP_QUERY_FLAG_NUMBER = 0x20000000;
constexpr DWORD ICU_ESCAPE = 0x80000000;
typedef WORD INTERNET_PORT;
typedef enum { INTERNET_SCHEME_HTTP = 1, INTERNET_SCHEME_HTTPS = 2 } INTERNET_SCHEME;
typedef struct _URL_COMPONENTS {
    DWORD dwStructSize; LPWSTR lpszScheme; DWORD dwSchemeLength; INTERNET_SCHEME nScheme; LPWSTR lpszHostName; DWORD dwHostNameLength;
    INTERNET_PORT nPort; LPWSTR lpszUserName; DWORD dwUserNameLength; LPWSTR lpszPassword; DWORD dwPasswordLength;
    LPWSTR lpszUrlPath; DWORD dwUrlPathLength; LPWSTR lpszExtraInfo; DWORD dwExtraInfoLength;
} URL_COMPONENTS, *LPURL_COMPONENTS;
#define WINHTTP_NO_PROXY_NAME nullptr
#define WINHTTP_NO_PROXY_BYPASS nullptr
#define WINHTTP_NO_REFERER nullptr
#define WINHTTP_DEFAULT_ACCEPT_TYPES nullptr
#define WINHTTP_NO_ADDITIONAL_HEADERS nullptr
#define WINHTTP_NO_REQUEST_DATA nullptr
#define WINHTTP_HEADER_NAME_BY_INDEX nullptr
#define WINHTTP_NO_HEADER_INDEX nullptr
HINTERNET WinHttpOpen(LPCWSTR agent, DWORD access, LPCWSTR proxy, LPCWSTR bypass, DWORD flags);
BOOL WinHttpSetTimeouts(HINTERNET h, int resolve, int connect, int send, int receive);
BOOL WinHttpSetOption(HINTERNET h, DWORD option, LPVOID buf, DWORD size);
BOOL WinHttpCrackUrl(LPCWSTR url, DWORD length, DWORD flags, LPURL_COMPONENTS uc);
HINTERNET WinHttpConnect(HINTERNET session, LPCWSTR host, INTERNET_PORT port, DWORD reserved);
HINTERNET WinHttpOpenRequest(HINTERNET conn, LPCWSTR verb, LPCWSTR path, LPCWSTR version, LPCWSTR referer, LPCWSTR* accept, DWORD flags);
BOOL WinHttpSendRequest(HINTERNET req, LPCWSTR headers, DWORD headersLen, LPVOID data, DWORD dataLen, DWORD totalLen, DWORD_PTR ctx);
BOOL WinHttpReceiveResponse(HINTERNET req, LPVOID reserved);
BOOL WinHttpQueryHeaders(HINTERNET req, DWORD level, LPCWSTR name, LPVOID buf, LPDWORD len, LPDWORD index);
BOOL WinHttpReadData(HINTERNET req, LPVOID buf, DWORD size, LPDWORD got);
BOOL WinHttpCloseHandle(HINTERNET h);

// ---- OS version (RtlGetVersion is reached through GetProcAddress of ntdll, which does not load here) ----------------------
typedef HANDLE HRSRC;
typedef HANDLE HGLOBAL;
#define MAKEINTRESOURCEW(i) ((LPWSTR)(ULONG_PTR)(WORD)(i))
HRSRC FindResourceW(HMODULE m, LPCWSTR name, LPCWSTR type);   // RCDATA from the generated table (resources.S); fails when the build has none
HGLOBAL LoadResource(HMODULE m, HRSRC r);
DWORD SizeofResource(HMODULE m, HRSRC r);
LPVOID LockResource(HGLOBAL h);
BOOL QueryFullProcessImageNameW(HANDLE process, DWORD flags, LPWSTR out, LPDWORD size);
inline int gmtime_s(struct tm* out, const time_t* t) { return gmtime_r(t, out) ? 0 : 22; }   // the MSVC argument order

typedef struct _OSVERSIONINFOW {
    DWORD dwOSVersionInfoSize, dwMajorVersion, dwMinorVersion, dwBuildNumber, dwPlatformId; wchar_t szCSDVersion[128];
} OSVERSIONINFOW;
