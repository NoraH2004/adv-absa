#!/bin/bash

mkdir -p ~/.local
cp -r share ~/.local/
update-mime-database ~/.local/share/mime
update-desktop-database ~/.local/share/applications
