; StudioForge -- Inno Setup 6 installer SKELETON (not built in CI yet).
;
; Status: structure and decisions only. Nothing here has been compiled; the
; "build the payload" step it depends on is described in README.md next to this
; file. Replace every TODO before shipping.
;
; Build (on a Windows runner with Inno Setup 6 installed):
;   iscc /DAppVersion=2.0.0 /DPayloadDir=..\..\dist\windows\payload deploy\windows\studioforge.iss
;
; What the installer does
;   1. copies the payload (embedded Python + the venv built from the release
;      wheel, plus WinSW and the tray launcher) to {autopf}\StudioForge;
;   2. registers a Windows SERVICE "StudioForge" running `studioforge serve`
;      (WinSW wrapper -- see README.md for why a wrapper and not pywin32);
;   3. adds a Start-menu entry and a per-user Startup shortcut for the TRAY
;      app, which shows status + base URL and has an "Open panel" action;
;   4. opens firewall rules for 1234/8080/1235 ONLY if the user ticks the LAN task;
;   5. on uninstall: stops/removes the service, removes the program files and
;      KEEPS the data directory (%LOCALAPPDATA%\studioforge) unless the user
;      ticks "delete my data" on the uninstall page.

#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif
#ifndef PayloadDir
  #define PayloadDir "..\..\dist\windows\payload"
#endif

#define AppName "StudioForge"
#define AppPublisher "LaserLloyd"
#define AppURL "https://github.com/LaserLloyd/StudioForge"

[Setup]
; TODO: generate once with Inno's GUID tool and never change it.
AppId={{00000000-0000-0000-0000-000000000000}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
DefaultDirName={autopf}\StudioForge
DefaultGroupName=StudioForge
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputBaseFilename=StudioForge-{#AppVersion}-setup
OutputDir=..\..\dist\windows
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName={#AppName}
; The service and the tray must not be running while files are replaced.
CloseApplications=yes
RestartApplications=no
; Authenticode: runs only when the owner's certificate secret exists (dormant
; otherwise). The release workflow passes /Ssigntool=... on the iscc command line.
; SignTool=signtool
; SignedUninstaller=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "service"; Description: "Run StudioForge as a Windows service (starts at boot, before login)"; GroupDescription: "Startup:"; Flags: unchecked
Name: "tray"; Description: "Start the tray app at login (status, base URL, Open panel)"; GroupDescription: "Startup:"
Name: "lan"; Description: "Allow other machines on my network to connect (opens ports 1234, 8080, 1235 in Windows Firewall, private networks only)"; GroupDescription: "Network:"; Flags: unchecked

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\StudioForge panel"; Filename: "{app}\studioforge-open.cmd"; IconFilename: "{app}\studioforge.ico"
Name: "{group}\StudioForge tray"; Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: "-m studioforge tray"
Name: "{userstartup}\StudioForge tray"; Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: "-m studioforge tray"; Tasks: tray

[Run]
; WinSW: <name>.exe + <name>.xml side by side; the XML is shipped in the payload.
Filename: "{app}\winsw\StudioForgeSvc.exe"; Parameters: "install"; Flags: runhidden; Tasks: service; StatusMsg: "Registering the StudioForge service..."
Filename: "{app}\winsw\StudioForgeSvc.exe"; Parameters: "start"; Flags: runhidden; Tasks: service; StatusMsg: "Starting the StudioForge service..."
Filename: "netsh"; Parameters: "advfirewall firewall add rule name=""StudioForge"" dir=in action=allow protocol=TCP localport=1234,8080,1235 profile=private"; Flags: runhidden; Tasks: lan
Filename: "{app}\studioforge-open.cmd"; Description: "Open the control panel"; Flags: postinstall nowait skipifsilent

[UninstallRun]
Filename: "{app}\winsw\StudioForgeSvc.exe"; Parameters: "stop"; Flags: runhidden; RunOnceId: "SvcStop"
Filename: "{app}\winsw\StudioForgeSvc.exe"; Parameters: "uninstall"; Flags: runhidden; RunOnceId: "SvcRemove"
Filename: "netsh"; Parameters: "advfirewall firewall delete rule name=""StudioForge"""; Flags: runhidden; RunOnceId: "FwRemove"

[Code]
// "Delete my data" checkbox on the uninstall progress page. DEFAULT: keep it --
// the data directory holds config.yaml, the model registry, built engines and
// logs, and an uninstall/reinstall must not cost anyone a long engine build.
var
  DeleteDataBox: TNewCheckBox;

procedure InitializeUninstallProgressForm();
begin
  DeleteDataBox := TNewCheckBox.Create(UninstallProgressForm);
  DeleteDataBox.Parent := UninstallProgressForm.InnerPage;
  DeleteDataBox.Caption := 'Also delete my StudioForge data (settings, registry, engines, logs)';
  DeleteDataBox.Checked := False;
  DeleteDataBox.Left := 0;
  DeleteDataBox.Top := UninstallProgressForm.StatusLabel.Top + 40;
  DeleteDataBox.Width := UninstallProgressForm.InnerPage.Width;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if (CurUninstallStep = usPostUninstall) and DeleteDataBox.Checked then
    DelTree(ExpandConstant('{localappdata}\studioforge'), True, True, True);
end;
