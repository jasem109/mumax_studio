# Why downloads were flagged, and how to stop it

Browsers and Windows judge a downloaded program by two things: whether it is **signed** by a known publisher,
and how many people have already downloaded that exact file (**reputation**). A new, unsigned installer has
neither, so Microsoft Edge says "not commonly downloaded", Chrome may say "can harm your computer", and
SmartScreen shows "Windows protected your PC". Some antivirus engines also flag programs built with
PyInstaller, because the prebuilt PyInstaller loader is shared with malware built the same way.

## Already done in the build

| Measure | Where | Effect |
| --- | --- | --- |
| Folder build instead of a one-file self-extractor | `build_exe.py` | one-file executables unpack code at run time, which heuristics dislike |
| No UPX compression (`--noupx`) | `build_exe.py` | UPX-packed files are a top false-positive cause |
| Bootloader compiled in CI from source | `release.yml` | the file no longer matches signatures of other PyInstaller programs |
| Windows version resource (company, product, description) | `build_exe.py` | programs without it look like throw-away droppers |
| `SHA256SUMS.txt` published with every release | `release.yml` | users can verify the file they got is the one that was built |
| Signing step, active as soon as secrets exist | `release.yml` | see below |
| Installer-free route: `pip install` from GitHub | README | nothing for SmartScreen to judge |

## What only the repository owner can do

1. **Get a code-signing identity.** The cheapest option for an individual is
   [Azure Trusted Signing](https://learn.microsoft.com/azure/trusted-signing/) (about USD 10 per month, identity
   validation for individuals is available in many countries). Open-source projects can apply to the free
   [SignPath Foundation](https://signpath.org/) programme instead (then replace the two signing steps in
   `release.yml` with the SignPath action).
2. For Azure Trusted Signing, create an app registration with the *Trusted Signing Certificate Profile Signer*
   role and add these repository secrets (GitHub → Settings → Secrets and variables → Actions):
   `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_SIGNING_ENDPOINT`
   (e.g. `https://weu.codesigning.azure.net/`), `AZURE_SIGNING_ACCOUNT`, `AZURE_CERT_PROFILE`.
   The next tagged release is signed automatically.
3. **Report false positives** for each new release: submit the installer at
   <https://www.microsoft.com/wdsi/filesubmission> (choose "Software developer", "Incorrectly detected") and,
   if a specific engine flags it, check <https://www.virustotal.com> and use that vendor's false-positive form.
4. Unsigned files earn reputation per file, so every new release starts from zero. Once releases are signed,
   reputation is tied to the signing identity and carries over from one release to the next.
