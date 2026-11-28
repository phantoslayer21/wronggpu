// SPDX-License-Identifier: GPL-3.0-only
#pragma once

namespace wg {

// Embedded RCDATA 960 from the published 0.6.0 and 0.6.5 releases, and the local RC2 release.
// The 0.5.5 RC1 local release has separate hosted and direct restorers without version resources.
inline constexpr const wchar_t* kKnownRestorerShas[] = {
    L"d642261731e54cb32aa15a3a3343d92d1532f3bb89add1637d77b66c33209571",
    L"71aac75a74271a0dc75ae81bc9cea460864a96481e68d733ca1e4f05ac69f350",
    L"481b586ef789b961263f7e8a13c52759ed748e13ab9280a4ece95a11285f9616",
    L"6ca02b4f90ba4b8ccb47deb78d0ce94eed636151535dffa2104a838eb778bb8a",
    L"916027355622d55d1fd6a4bb6151eb2eebd9989aa11a613ae57cd4fae33a42b1",
};

}  // namespace wg
