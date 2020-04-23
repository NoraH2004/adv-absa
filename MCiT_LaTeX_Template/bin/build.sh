# Build script for zip files

## Create zip files
cd templates/

## Article
cd Article/
rm ../Article.zip
zip --symlinks -r ../Article.zip . --exclude "*.aux" "*.fdb_latexmk" "*.fls" "*.log" "*.synctex*"  "*.toc" "*.bbl" "*.blg" "*.lof" "*.lot"
cd ../

## Expose
cd Expose/
rm ../Expose.zip
zip --symlinks -r ../Expose.zip . --exclude "*.aux" "*.fdb_latexmk" "*.fls" "*.log" "*.synctex*"  "*.toc" "*.bbl" "*.blg" "*.lof" "*.lot"
cd ..

## Thesis
cd Thesis/
rm ../Thesis.zip
zip --symlinks -r ../Thesis.zip . --exclude "*.aux" "*.fdb_latexmk" "*.fls" "*.log" "*.synctex*"  "*.toc" "*.bbl" "*.blg" "*.lof" "*.lot"