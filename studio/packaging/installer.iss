; Inno Setup script: turns dist\mumax-studio\ (from build_exe.py) into a normal Windows installer
; with Start-menu / desktop shortcuts, an uninstaller and an "Open with mumax Studio" entry for .mx3 files.
;   iscc /DAppVersion=0.5.3 packaging\installer.iss      -> dist\mumax-studio-windows-setup.exe
; The installer asks where to install and whether simulations run on the CPU or an NVIDIA GPU. For the GPU it
; downloads the CUDA libraries listed in build\gpu-wheels.txt (written by build_exe.py) and lets the app unpack
; them into %USERPROFILE%\.mumaxstudio\gpu (see mumaxstudio\gpupack.py).

#ifndef AppVersion
  #define AppVersion "0.5.3"
#endif
; Python version of the build: the GPU folder is per Python version (gpupack.tag())
#ifndef PyTag
  #define PyTag "cp312"
#endif

[Setup]
AppId={{6C2F3D4E-8B1A-4F7E-9C55-2A7B5D1E9F31}
AppName=mumax Studio
AppVersion={#AppVersion}
AppPublisher=Mohammad Abu Jasem
AppPublisherURL=https://majasem.com/mumaxUI
DefaultDirName={autopf}\mumax Studio
DefaultGroupName=mumax Studio
DisableProgramGroupPage=yes
; always ask where to install (also when updating by hand; in-app updates run silently)
DisableDirPage=no
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=mumax-studio-windows-setup
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\mumax-studio.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesAssociations=yes
; the app quits by itself before an in-app update and the [Run] entry below restarts it; letting the Restart
; Manager restart it too would start two copies, racing the file copy
CloseApplications=yes
RestartApplications=no
LicenseFile=..\LICENSE

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "assoc"; Description: "Open .mx3 scripts with mumax Studio"; GroupDescription: "Files:"

[Files]
Source: "..\dist\mumax-studio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; read by the device page below, not installed (the app has its own copy)
Source: "..\build\gpu-wheels.txt"; Flags: dontcopy skipifsourcedoesntexist

[Icons]
Name: "{group}\mumax Studio"; Filename: "{app}\mumax-studio.exe"
Name: "{group}\Uninstall mumax Studio"; Filename: "{uninstallexe}"
Name: "{autodesktop}\mumax Studio"; Filename: "{app}\mumax-studio.exe"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.mx3"; ValueType: string; ValueName: ""; ValueData: "mumaxStudio.mx3"; Flags: uninsdeletevalue; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\mumaxStudio.mx3"; ValueType: string; ValueName: ""; ValueData: "mumax3 script"; Flags: uninsdeletekey; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\mumaxStudio.mx3\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\mumax-studio.exe,0"; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\mumaxStudio.mx3\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\mumax-studio.exe"" ""%1"""; Tasks: assoc

[InstallDelete]
; CuPy and the CUDA libraries that the former NVIDIA edition (up to 0.5.2) bundled: left in place they would be
; imported instead of the downloaded GPU support
Type: filesandordirs; Name: "{app}\_internal\cupy"
Type: filesandordirs; Name: "{app}\_internal\cupy_backends"
Type: filesandordirs; Name: "{app}\_internal\cupyx"
Type: filesandordirs; Name: "{app}\_internal\fastrlock"
Type: filesandordirs; Name: "{app}\_internal\nvidia"
Type: filesandordirs; Name: "{app}\_internal\cuda"
Type: filesandordirs; Name: "{app}\_internal\cuda_pathfinder"

[UninstallDelete]
; the downloaded GPU support
Type: filesandordirs; Name: "{%USERPROFILE}\.mumaxstudio\gpu"

[Run]
Filename: "{app}\mumax-studio.exe"; Description: "Start mumax Studio"; Flags: nowait postinstall skipifsilent
; an in-app update runs this installer silently: start the new version afterwards, as the user (not elevated)
Filename: "{app}\mumax-studio.exe"; Flags: nowait runasoriginaluser; Check: WizardSilent

[Code]
var
  DevicePage: TInputOptionWizardPage;
  DownloadPage: TDownloadWizardPage;
  Wheels: TArrayOfString;
  GpuBytes: Int64;
  Downloaded: Boolean;
  WasNvidiaEdition: Boolean;

function Token(S: String; N: Integer): String;
{ the N-th (0-based) space separated field of S }
var
  i, p: Integer;
begin
  S := Trim(S);
  for i := 1 to N do
  begin
    p := Pos(' ', S);
    if p = 0 then
    begin
      Result := '';
      exit;
    end;
    S := Trim(Copy(S, p + 1, Length(S)));
  end;
  p := Pos(' ', S);
  if p > 0 then
    Result := Copy(S, 1, p - 1)
  else
    Result := S;
end;

function IsWheelLine(S: String): Boolean;
begin
  Result := (Length(Trim(S)) > 0) and (Copy(Trim(S), 1, 1) <> '#') and (Token(S, 3) <> '');
end;

function NvidiaPresent: Boolean;
begin
  { nvcuda.dll comes with every NVIDIA graphics driver }
  Result := FileExists(ExpandConstant('{sys}\nvcuda.dll'));
end;

function GpuReady: Boolean;
begin
  Result := FileExists(ExpandConstant('{%USERPROFILE}\.mumaxstudio\gpu\{#PyTag}-win\READY'));
end;

function UseGpu: Boolean;
begin
  Result := (DevicePage <> nil) and (DevicePage.SelectedValueIndex = 1);
end;

procedure InitializeWizard;
var
  i: Integer;
  GpuText: String;
begin
  GpuBytes := 0;
  Downloaded := False;
  try
    ExtractTemporaryFile('gpu-wheels.txt');
  except
  end;
  if LoadStringsFromFile(ExpandConstant('{tmp}\gpu-wheels.txt'), Wheels) then
    for i := 0 to GetArrayLength(Wheels) - 1 do
      if IsWheelLine(Wheels[i]) then
        GpuBytes := GpuBytes + StrToInt64Def(Token(Wheels[i], 3), 0);

  DevicePage := CreateInputOptionPage(wpSelectDir, 'Simulation device',
    'Where should mumax Studio run simulations?',
    'The GPU is much faster for large meshes. You can switch between them at any time in the Device menu of the app.',
    True, False);
  DevicePage.Add('CPU (processor): works on every computer');
  if not NvidiaPresent then
    GpuText := 'NVIDIA GPU: not available (no NVIDIA graphics driver found on this computer)'
  else if GpuReady then
    GpuText := 'NVIDIA GPU (GPU support is already installed)'
  else if GpuBytes > 0 then
    GpuText := 'NVIDIA GPU: downloads the CUDA libraries now (about ' + IntToStr(GpuBytes div 1000000) + ' MB)'
  else
    GpuText := 'NVIDIA GPU: not available in this build';
  DevicePage.Add(GpuText);
  DevicePage.CheckListBox.ItemEnabled[1] := NvidiaPresent and (GpuReady or (GpuBytes > 0));
  if DevicePage.CheckListBox.ItemEnabled[1] then
    DevicePage.SelectedValueIndex := 1
  else
    DevicePage.SelectedValueIndex := 0;

  DownloadPage := CreateDownloadPage('Downloading GPU support', 'The CUDA libraries for NVIDIA GPUs are being downloaded.', nil);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  i: Integer;
begin
  Result := True;
  if (CurPageID = wpReady) and UseGpu and not GpuReady and not Downloaded then
  begin
    DownloadPage.Clear;
    for i := 0 to GetArrayLength(Wheels) - 1 do
      if IsWheelLine(Wheels[i]) then
        DownloadPage.Add(Token(Wheels[i], 1), Token(Wheels[i], 0), Token(Wheels[i], 2));
    DownloadPage.Show;
    try
      try
        DownloadPage.Download;
        Downloaded := True;
      except
        if DownloadPage.AbortedByUser then
          Result := False
        else if SuppressibleMsgBox('The GPU support could not be downloaded:' + #13#10 + GetExceptionMessage + #13#10#13#10 +
            'Install anyway and run simulations on the CPU? (You can add GPU support later by running this installer again.)',
            mbConfirmation, MB_YESNO, IDYES) = IDYES then
          DevicePage.SelectedValueIndex := 0
        else
          Result := False;
      end;
    finally
      DownloadPage.Hide;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Params: String;
  Code: Integer;
begin
  if CurStep = ssInstall then
    { before [InstallDelete] removes it: did the former NVIDIA edition (bundled CuPy) live here? }
    WasNvidiaEdition := DirExists(ExpandConstant('{app}\_internal\cupy'));
  if CurStep <> ssPostInstall then
    exit;
  if WizardSilent then
  begin
    { in-app update: a former NVIDIA edition keeps its GPU by downloading the GPU support now }
    if WasNvidiaEdition and NvidiaPresent and not GpuReady then
      ExecAsOriginalUser(ExpandConstant('{app}\mumax-studio.exe'), 'gpu-setup', '', SW_HIDE, ewWaitUntilTerminated, Code);
    exit;
  end;
  if UseGpu and Downloaded then
  begin
    WizardForm.StatusLabel.Caption := 'Setting up GPU support...';
    Params := 'gpu-setup --default --from "' + ExpandConstant('{tmp}') + '"';
  end
  else if UseGpu then
    Params := 'gpu-setup --use gpu'
  else
    Params := 'gpu-setup --use cpu';
  if not ExecAsOriginalUser(ExpandConstant('{app}\mumax-studio.exe'), Params, '', SW_HIDE, ewWaitUntilTerminated, Code) or (Code <> 0) then
    if UseGpu then
      SuppressibleMsgBox('GPU support could not be set up, so mumax Studio will use the CPU.' + #13#10 +
        'Run this installer again to retry.', mbError, MB_OK, IDOK);
end;
