// SPDX-License-Identifier: MIT
// Linux: stands in for the Windows header of the same name (win32_posix_rm.cpp). The Restart Manager's "which processes
// hold these files" on /proc: a process holds a file when one of its open files (/proc/<pid>/fd) or mapped files
// (/proc/<pid>/maps: a library, or a DLL a Wine process has loaded) is that file (same device and inode). Only this
// user's processes are looked at (another user's fd and maps are closed to it): a file another account's program holds is
// not listed. A process of this user whose fd or maps cannot be read (not dumpable, or holding capabilities: on a desktop
// the compositor and ssh-agent, for example) adds RmRebootReasonPermissionDenied to the reasons, so on most desktops that
// reason is always there; an empty list means "none seen", never more. Restarting or shutting down applications is not available.
#pragma once
#include "win32_posix.h"

constexpr DWORD CCH_RM_SESSION_KEY = 32;   // sizeof(GUID) * 2
constexpr DWORD CCH_RM_MAX_APP_NAME = 255;
constexpr DWORD CCH_RM_MAX_SVC_NAME = 63;
typedef enum _RM_APP_TYPE { RmUnknownApp = 0, RmMainWindow = 1, RmOtherWindow = 2, RmService = 3, RmExplorer = 4, RmConsole = 5, RmCritical = 1000 } RM_APP_TYPE;
typedef enum _RM_REBOOT_REASON {
    RmRebootReasonNone = 0x0, RmRebootReasonPermissionDenied = 0x1, RmRebootReasonSessionMismatch = 0x2, RmRebootReasonCriticalProcess = 0x4,
    RmRebootReasonCriticalService = 0x8, RmRebootReasonDetectedSelf = 0x10
} RM_REBOOT_REASON;
typedef struct _RM_UNIQUE_PROCESS { DWORD dwProcessId; FILETIME ProcessStartTime; } RM_UNIQUE_PROCESS;
typedef struct _RM_PROCESS_INFO {
    RM_UNIQUE_PROCESS Process;   // ProcessStartTime: the process's start (from /proc/<pid>/stat)
    WCHAR strAppName[CCH_RM_MAX_APP_NAME + 1];   // the program's file name (a Wine process: its Windows program's name)
    WCHAR strServiceShortName[CCH_RM_MAX_SVC_NAME + 1];   // always empty
    RM_APP_TYPE ApplicationType;   // always RmUnknownApp
    ULONG AppStatus;
    DWORD TSSessionId;
    BOOL bRestartable;   // always FALSE
} RM_PROCESS_INFO;

DWORD RmStartSession(DWORD* session, DWORD flags, WCHAR key[]);
// files only: applications or services to register are ERROR_NOT_SUPPORTED
DWORD RmRegisterResources(DWORD session, UINT nFiles, LPCWSTR files[], UINT nApplications, RM_UNIQUE_PROCESS applications[], UINT nServices,
                          LPCWSTR services[]);
// ERROR_MORE_DATA (and *needed) when *count is too small; *reasons as above
DWORD RmGetList(DWORD session, UINT* needed, UINT* count, RM_PROCESS_INFO info[], LPDWORD reasons);
DWORD RmEndSession(DWORD session);
