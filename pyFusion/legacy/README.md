# Image Processing and Fusion Toolbox
![license](https://img.shields.io/bower/l/bootstrap.svg?color=blue) <a href="https://996.icu/#/en_US"><img src="https://img.shields.io/badge/link-996.icu-red.svg" alt="996.icu" /></a> <a href="https://1989Ryan.github.io/DIPhw/diphw.html"><img src="https://img.shields.io/badge/link-DIP%20homework-green.svg"  /></a>

XJTU undergraduate course, Digital Image and Video Processing and Multi-data Fusion. You can visit my [homework website](https://1989Ryan.github.io/DIPhw/diphw.html) to see my DIP homework report and projects.

## basic cv toolbox for python

The basic computer vision toolbox for python, which consists of some basic tools for computer vision and image fusion. Most tools are not covered by opencv. 

This toolbox is only used for DIP course and Multi-Sensor Fusion course.

### prerequisite

Although no one cares a shit about that project, this toolbox needs python-opencv.

### Structure

```
CV_Python_Toolbox\
  scripts\
    histogram.py
    filter.py
    FreqFilter.py
  src\
    __init__.py
    basic_cv_tool.py
    image_fusion_tool.py
  test\
    *
homework1\
homework2\
homwwork3\
homework4\
homework5\
homework6\
.gitignore
README.md
LICENSE
```

## homework1

First project assignment report, using latex.

If you want to use my paper structure, you have to install a considerable number of packages for tex. So forget it.

## homework2

Second project assignment report. We have updated the toolbox for some new functions.

## homework3

Third project assignment supporting materials.

The toolbox has been updated to the latest version which contains all the tools for image histogram equalization, histogram specialization, local histogram transformation and image segmentation using histogram thresholding.

## homework4

Fourth project assignment supporting materials.

The toolbox has been updated to the latest version which contains all the basic tools for spacial filtering including gaussian filter and high-pass filter module. You can use it directly or see my script ``CV_Python_Toolbox\scripts\filter.py`` to learn how to use it.

## homework5

Fifth project assignment supporting materials.

The toolbox has been updated to the latest version which contains all the basic tools for frequency domain filtering including BLPF, GLPF, BHPF, GHPF, Laplacian and Unsharp Masking. You can see my script ``CV_Python_Toolbox\scripts\FreqFilter.py``  or visit my [homework report](https://1989Ryan.github.io/DIPhw/hw5.html) to learn how to use it.

## homework6

Sixth project assignment supporting materials.

The toolbox has been updated to the latest version which contains all the basic tools for image restoration and reconstruction including mean filtering, order-statistics filtering, Wiener fitering and constrained least squares filtering. You can visit my [homework report](https://1989Ryan.github.io/DIPhw/hw6.html) to see how to use them.

## New Updates!

I just update the toolbox with image fusion tool. I just add several methods for image fusion which is simple and not convenient. You can use it for fun since there are several bugs :-). This toolbox update is just aimed at finishing my project of multi-data fusion course.
