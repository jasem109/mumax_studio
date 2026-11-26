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
echo "Installed. Start it from the applications menu or run: mumax-studio"
