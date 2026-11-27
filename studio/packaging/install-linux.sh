#!/usr/bin/env sh
# Installs the unpacked mumax Studio folder for the current user: ~/.local/opt, a launcher and a menu entry.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
dest="$HOME/.local/opt/mumax-studio"
mkdir -p "$HOME/.local/opt" "$HOME/.local/bin" "$HOME/.local/share/applications" "$HOME/.local/share/icons/hicolor/512x512/apps"
rm -rf "$dest"
cp -r "$here/mumax-studio" "$dest"
ln -sf "$dest/mumax-studio" "$HOME/.local/bin/mumax-studio"
cp "$here/icon.png" "$HOME/.local/share/icons/hicolor/512x512/apps/mumax-studio.png"
sed "s|^Exec=.*|Exec=$dest/mumax-studio gui %f|" "$here/mumax-studio.desktop" > "$HOME/.local/share/applications/mumax-studio.desktop"
# optional NVIDIA GPU support (~0.5 GB download into ~/.mumaxstudio/gpu; see mumaxstudio/gpupack.py)
if [ -e /proc/driver/nvidia/version ] || command -v nvidia-smi >/dev/null 2>&1; then
  ans="${MUMAX_STUDIO_GPU:-}"
  if [ -z "$ans" ] && [ -t 0 ]; then
    printf "NVIDIA GPU found. Download GPU support now (about 0.5 GB)? [Y/n] "
    read -r ans || ans=n
    [ -z "$ans" ] && ans=y
  fi
  case "$ans" in
    [Yy]*) "$dest/mumax-studio" gpu-setup --default || echo "GPU support could not be installed; simulations run on the CPU. Retry with: mumax-studio gpu-setup --default" ;;
    *) echo "Skipped. Add GPU support later with: mumax-studio gpu-setup --default" ;;
  esac
fi
echo "Installed. Start it from the applications menu or run: mumax-studio"
