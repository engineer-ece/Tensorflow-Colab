# Matplotlib


## 1. Introduction
   ### 1. [Usage Guide]()
   1. [A simple example]()
   2. [Parts of a Figure]()
      1. [Figure]()
      2. [Axes]()
      3. [Axis]()
      4. [Artist]()
   3. [Types of inputs to plotting functions]()
   4. [The object-oriented interface and the pyplot interface]()
   5. [Backends]()
      1. [What is a backend?]()
      2. [Selecting a backend]()
      3. [The builtin backend]()
        1. [ipympl]()
        2. [How do I select PyQt4 or PySide?]()
      4. [Using non-builtin backends]()
   6. [What is interactive mode?]()
      1. [Interactive example]()
      2. [Non-interactive example]()
      3. [Summary]()
   7. [Performance]()
      1. [Line segment simplification]()
      2. [Marker simplification]()
      3. [Splitting lines into smaller chunks]()
      4. [Legends]()
      5. [Using the fast style]()
   
   ### 2. Pyplot 
   1. [Intro to pyplot]()
      1. [Formatting the style of your plot]()
   2. [Plotting with keyword strings]()
   3. [Plotting with categorical variables]()
   4. [Controlling line properties]()
   5. [Working with multiple figures and axes]()
   6. [Working with text]()
      1. [Using mathematical expressions in text]()
      2. [Annotating text]()
   7. [Logarithmic and other nonlinear axes]()
   
   ### 3. Sample plots
   1. Line plot
   2. Multiple subplots in one figure
   3. Images
   4. Contouring and pseudocolor
   5. Histograms
   6. Paths
   7. Three-dimensional plotting
   8. Streamplot
   9. Ellipses
   10. Bar charts
   11. Pie charts
   12. Tables
   13. Scatter plots
   14. GUI widgets
   15. Filled curves
   16. Date handling
   17. Log plots
   18. Polor plots
   19. Legends
   20. TeX-notation for text objects
   21. Native TeX rendering
   22. EEG GUI
   23. XKCD-style sketch plots
   24. Subplot example
   ### 4. Image 
   1. Startup commands
   2. Importing image data into Numpy array
   3. Plotting numpy arrays as images
      1. Applying pseudocolor schemes to image plots
      2. Color scale reference
      3. Examining a specific data range
      4. Array Interpolation schemes
   ### 5. Lifecycle of Plot
   1. A note on the Object-Oriented API vs Pyplot
   2. Our data
   3. Getting started
   4. Controlling the style
   5. Customizing the plot
   6. Combining multiple visualizations
   7. Saving out plot
   ### 6. Customizing Matplotlib (style sheet & rcParams)
   * sheets   
     1. Using style sheets
     2. Defining your own style
     3. Composing styles
     4. Temporary styling
   * Matplotlib rcParams
     1. Dynamic rc settings
     2. The matplotlibrc file
       1. A sample matplotlibrc file
## 2. Intermediate
   ### 1. Artist tutorial
   1. Customizing your objects
   2. Object containers
    1. Figure container
    2. Axes container
    3. Axis container
    4. Tick container
   ### 2. Legend guide
   1. Controlling the legend entries
   2. Creating artists specifically for adding to the legend(aka,Proxy artists)
   3. Legend location
   4. Multiple legends on the same Axes
   5. Legend Handlers
      1. Implementing a custom legend handler
   ### 3. Styling with cycler
   1. Setting prop_cycle in the matplotlibrc or style files
   2. Cycling through multiple properties
   ### 4. Customizing Figure Layouts Using GridSpec & Other Function
   1. Basic Quickstart Guide
   2. Fine Adjustment to Gridspec Layout
   3. GridSpec using SubplotSpec
   4. A Complex Nested GridSpec using SubplotSpec
   ### 5. Constrained Layout Guide
   1. Simple Example
   2. Colorbars
   3. Suptitle
   4. Legends
   5. Padding and Spacing
     1. Spacing with colorbars
   6. rcParams
   7. Use with GridSpec
   8. Manually setting axes positions
   9. Manually turning off constrained_layout
   10. Limitations
      1. Incompatible functions
      2. Other Caveats
   11. Debugging
   12. Notes on the algorithm
      1. Figure layout
      2. Simple case: One Axes
      3. Simple case: two Axes
      4. Two Axes and colorbar
      5. Colorbar associated with a Gridspec
      6. Uneven sized Axes
      7. Empty gridspec slots
      8. Other notes
   ### 6. Tight Layout guide
   1. Simple Example
   2. Caveats
   3. Use with GridSpec
   4. Legends and Annotations
   5. Use with AxesGrid1
   6. Colorbar
   ### 7. Origin and extent in imshow
   1. Default extent
   2. Explicit extent
   3. Explicit extent and axes limits
## 3. Advanced
   ### 1. Blitting tutorial
   ### 2. Path Tutorial
   ### 3. Path effects guide
   ### 4. Transformations Tutorial
## 4. Colors
   ### 1. Specifying Colors
   ### 2. Customized Colorbars 
   ### 3. Creating Colormaps
   ### 4. Colormap Normalization
   ### 5. Choosing Colormaps
## 5. Provisional 
   ### 1. Complex and semantic figure composition
## 6. Text
   ### 1. Text 
   ### 2. Text properties and layout
   ### 3. Annotation
   ### 4. Writing mathematical expressions
   ### 5. Typesetting with XeLaTeX/LuaLaTex
   ### 6. Text rendering with LaTeX
## 7. Toolkits
   ### 1. Overview of axes_grid1 toolkit
   ### 2. Overview of axisartist toolkit
   ### 3. The mplot3d Toolkit
