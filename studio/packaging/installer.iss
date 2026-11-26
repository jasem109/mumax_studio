; Inno Setup script: turns dist\mumax-studio\ (from build_exe.py) into a normal Windows installer
; with Start-menu / desktop shortcuts, an uninstaller and an "Open with mumax Studio" entry for .mx3 files.
;   iscc /DAppVersion=0.4.0 packaging\installer.iss      -> dist\mumax-studio-windows-setup.exe

#ifndef AppVersion
  #define AppVersion "0.4.0"
#endif
#ifndef Edition
  #define Edition ""
#endif

[Setup]
AppId={{6C2F3D4E-8B1A-4F7E-9C55-2A7B5D1E9F31}
AppName=mumax Studio
AppVersion={#AppVersion}
AppPublisher=mumax Studio contributors
AppPublisherURL=https://majasem.com/mumaxUI
DefaultDirName={autopf}\mumax Studio
DefaultGroupName=mumax Studio
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=mumax-studio-windows{#Edition}-setup
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\mumax-studio.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesAssociations=yes
LicenseFile=..\LICENSE

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "assoc"; Description: "Open .mx3 scripts with mumax Studio"; GroupDescription: "Files:"

[Files]
Source: "..\dist\mumax-studio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\mumax Studio"; Filename: "{app}\mumax-studio.exe"
Name: "{group}\Uninstall mumax Studio"; Filename: "{uninstallexe}"
Name: "{autodesktop}\mumax Studio"; Filename: "{app}\mumax-studio.exe"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.mx3"; ValueType: string; ValueName: ""; ValueData: "mumaxStudio.mx3"; Flags: uninsdeletevalue; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\mumaxStudio.mx3"; ValueType: string; ValueName: ""; ValueData: "mumax3 script"; Flags: uninsdeletekey; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\mumaxStudio.mx3\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\mumax-studio.exe,0"; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\mumaxStudio.mx3\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\mumax-studio.exe"" ""%1"""; Tasks: assoc

[Run]
Filename: "{app}\mumax-studio.exe"; Description: "Start mumax Studio"; Flags: nowait postinstall skipifsilent
; an in-app update runs this installer silently: start the new version afterwards, as the user (not elevated)
Filename: "{app}\mumax-studio.exe"; Flags: nowait runasoriginaluser; Check: WizardSilent
