// SPDX-License-Identifier: MIT
// CPU test of the e1b runtime option (clean/runtime/e1b_option.h): the value parse (on / off; every other value falls
// back to off with one log line, never a failed create), the scan of a whole options string, and the report item.
// Built and run by test_e1b_option.py.
#include <cstdio>
#include <string>

#include "../e1b_option.h"

static int failures = 0;

static void expect(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("FAIL %s\n", what.c_str());
    ++failures;
  }
}

static void value(const char* v, bool want_on, const std::string& want_note) {
  std::string note = "stale";
  const bool on = e1b_option_on(v, &note);
  expect(on == want_on && note == want_note,
         std::string("e1b=") + v + ": on " + (on ? "1" : "0") + ", note '" + note + "'");
}

static void scan(const char* opts, bool want_on, const std::string& want_note) {
  bool on = !want_on;
  std::string note = "stale";
  e1b_scan(opts, &on, &note);
  expect(on == want_on && note == want_note,
         std::string("scan '") + opts + "': on " + (on ? "1" : "0") + ", note '" + note + "'");
}

int main() {
  value("on", true, "");
  value("off", false, "");
  value("1", false, "e1b: unknown value '1', using off");
  value("ON", false, "e1b: unknown value 'ON', using off");
  value("yes", false, "e1b: unknown value 'yes', using off");
  value("", false, "e1b: unknown value '', using off");
  value("onn", false, "e1b: unknown value 'onn', using off");
  expect(!e1b_option_on("off", nullptr) && e1b_option_on("on", nullptr), "no note pointer");

  scan("", false, "");
  scan("executor=dx12;model=M", false, "");
  scan("executor=dx12;model=M;e1b=on", true, "");
  scan("e1b=on,model=M", true, "");
  scan("model=M;e1b=1", false, "e1b: unknown value '1', using off");
  scan("e1b=1;e1b=on", true, "");                        // the last e1b key decides
  scan("e1b=on;e1b=typo", false, "e1b: unknown value 'typo', using off");
  scan("e1bx=on;xe1b=on;e1b", false, "");               // other keys and a key without a value are ignored
  scan("e1b=on;", true, "");                             // the interop's "configuration;environment" join

  expect(e1b_report(true, "") == " e1b=on", "report on");
  expect(e1b_report(false, "Model K") == " e1b=off (Model K)", "report off");
  std::printf("e1b option: %d failure(s)\n", failures);
  return failures ? 1 : 0;
}
