%% mcitarticle.cls
%% 
%% Author: Daniel Mur-Spiegl <news@spiegl.me>
%% ------------------------------------------

%
%% Package Information
%% ------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mcitarticle}[2019/10/02]

%
%% Handling class options
%% ------------------------------------------

\DeclareOption{titlepage}{\PassOptionsToPackage{\CurrentOption}{mcit}}
\DeclareOption{english}{\PassOptionsToPackage{\CurrentOption}{mcit}}
\DeclareOption{german}{\PassOptionsToPackage{\CurrentOption}{mcit}}
\DeclareOption{master}{\PassOptionsToPackage{\CurrentOption}{mcit}}
\DeclareOption{bachelor}{\PassOptionsToPackage{\CurrentOption}{mcit}}
\DeclareOption*{
    \ClassWarning{mcitarticle}{Unknown document option '\CurrentOption'.}
}
\ProcessOptions

%
%% Paper options and style file -----------------------------------------------

\LoadClass[a4paper,12pt]{article}

\RequirePackage{mcit}       % mcit.sty with options registered above

%% Set title font to SF
%\RequirePackage{titling}
%\renewcommand{\maketitlehooka}{\sffamily}


% Add PDF meta data

%\AtBeginDocument{%
%	\hypersetup{
%		pdftitle={\@title},
%		pdfauthor={\@author},
%		pdfsubject={},
%		pdfcreator={LaTeX using class hgbarticle [\hgbDate]},
%		pdfproducer={pdflatex},
%		pdfkeywords={}
%	}%
%}%


%% Additional Hagenberg packages ----------------------------------------------

%\RequirePackage{hgbmath}
%\RequirePackage{hgbalgo}
%\RequirePackage{hgbheadings}
%\RequirePackage{hgbabbrev}
%\RequirePackage{hgblistings}
%\RequirePackage{hgbbib}

\endinput
