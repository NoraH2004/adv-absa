%% mcitthesis.cls
%% 
%% Author: Daniel Mur-Spiegl <news@spiegl.me>
%% ------------------------------------------

%
%% Package Information
%% ------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mcitthesis}[2019/05/30]

%
%% Handling class options
%% ------------------------------------------

\DeclareOption{titlepage}{\PassOptionsToPackage{\CurrentOption}{mcit}}
\DeclareOption{english}{\PassOptionsToPackage{\CurrentOption}{mcit}}
\DeclareOption{german}{\PassOptionsToPackage{\CurrentOption}{mcit}}
\DeclareOption{master}{\PassOptionsToPackage{\CurrentOption}{mcit}}
\DeclareOption{bachelor}{\PassOptionsToPackage{\CurrentOption}{mcit}}
\DeclareOption*{
    \ClassWarning{mcitthesis}{Unknown document option '\CurrentOption'.}
}
\ProcessOptions


%
%% Paper options and style file -----------------------------------------------

\LoadClass[a4paper,12pt]{article}

\RequirePackage{mcit}       % mcit.sty with options registered above


%
%% Commands
%% ------------------------------------------

\newcommand{\method}[1]{
    \newcommand{\themethod}{#1}
}

\newcommand{\advisor}[1]{
    \newcommand{\theadvisor}{#1}
}

\newcommand{\studentNumber}[1]{
    \newcommand{\thestudentNumber}{#1}
}

\newcommand{\academicYear}[1]{
    \newcommand{\theacademicYear}{#1}
}


\renewcommand{\maketitlepage}{
    \pagenumbering{Roman}

    \begin{titlepage}
        \newgeometry{top=3.5cm,bottom=2.5cm}
        \centering

        {\LARGE\bfseries\lsstyle \thetitle \par}
    
        \renewcommand{\baselinestretch}{1.1}

        \vspace{0.5cm}
        \vfill

        
        \fbox{\parbox[c][26pt][c]{\textwidth-6.8pt}{
            \centering {\LARGE\bfseries\scshape Master's Thesis}
            }} \par
        
        \vspace{0.5cm}
        \vfill
        
        Empirical Method\par
        {\bfseries \themethod}

        \vfill

        Master's Program\par
        {\large\bfseries "Management, Communication \& IT" \par}
        Management Center Innsbruck\par
    
        \vfill
        \vfill
        
        Thesis Advisor\par
        {\bfseries \theadvisor \par}
        
        \vfill

        Author \par
        {\bfseries Academic year \ifdefined\theacademicYear \theacademicYear \else \the\year \fi \par}
        {\bfseries \theauthor \par}
        {\bfseries \thestudentNumber \par}		
                        
        \vfill
                    
        % Bottom of the page
        {\ifdefined\thedate \thedate \else \today \fi\par}
        \restoregeometry
    \end{titlepage}   

    %% Table of Contents
    \tableofcontents
    \newpage

    %% Main Document
    \pagenumbering{arabic}
    \setcounter{page}{1} 
}

\newcommand{\makeaffidavit}{
    {\large\bfseries Affidavit}

    \vspace{0.5 cm}
    
    I hereby declare, under oath, that this Master’s Thesis has been my independent work and has not been aided with  any  prohibited  means.  I  declare,  to  the  best  of  my  knowledge  and  belief,  that  all  passages  taken  from  published  and  unpublished  sources  or  documents  have  been  reproduced  whether  as  original,  slightly  changed  or  in  thought,  have  been  mentioned  as  such  at  the  corresponding  places  of  the  thesis,  by  citation,  where the extent of the original quotes is indicated. 
    
    The paper has not been submitted for evaluation to another examination authority or has been published in this form or another.
    
    \vspace{1cm}
    
    Innsbruck, \ifdefined\thedate \thedate \else \today \fi
    
    \vspace{3cm}
    
    %
    %% Line for signature
    \rule{0.4\textwidth}{0.4pt}
    
    %
    %% Autor
    \theauthor
}



\endinput
