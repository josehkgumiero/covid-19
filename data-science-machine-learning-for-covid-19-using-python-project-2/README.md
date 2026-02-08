# Introduction

- Dash is an open-source Python framework used for building analytical web applications.
- It is a powerful library that simplifcaties the development of data-driven applications.
- Useful for Python dataa scientists who are not very familiar with web development.
- Users can create amazing dashbaords in their browser using dash.
- Built on top of Ploty.js, React, and Flask, Dash ties modern UI element like dropdown, sliders, and graphs directly to your analytical python code.
- Dash apps consist of a Flask server that communicates with front-end react components using JSON packets over HTTP requests.
- Dash applications are written purely in python, so no HMTL or Javascript is necessary.

# Dash app layout and components

Dash apps are composed of two parts.
1. First is the "layout" of the app which basically describes how the aplication looks like
2. The second part describes the interactivity of the application

# Dash Components
- Dash provides python classes of all the visual components of the application. We can also customize our own components with Javascript and React.js
- We can build the layout with dash_html_components and the dash_core_components library.
- The dash_html_components is for all html tags where the dash_core_components is for interactivity built with react.js

# Dash html components
- Instead of writing HTML or using an HTML templating engine, you compose your layout using Python structures with the dash-html-components library.

- we can find all the html tags like h1, div, p, header, etc.

# Dash core components
- dash ships with supercharged components for interactive user interfaces. A core set of components, written and maintaned by the dash team is available in the dash_core_components library.

# Adding Graphs using Plotly

Adding Graphs - Plotly

- the dash_core_components library includes a component called Graph.
- you can use this component graph to add figures to your dash app
- figures ca be generated using the plotly graphing library.
- the plotly python package existis to create, manipulate and render graphical figures (i.e. charts, plots, maps and diagrams) represented by data structurres also reffered to as figures

# Adding Graphps - Plotly

- The rendering process uses to Plotly.js, Javascript library under the hood although python developers using this module very rarely need to interact with the javascript librry directly, if ever.

- Plotly.js support around 35 chart types and renders charts in both vector -quality SVG and high-performance WEBGL.

- These are used in dcc.Graph with e.g. dcc.Graph(gigure=fig) with fig  a plotly figure.

# Ways of Generating figures using Plotly

## Method 1 - Figure as Dictionaries
- at  low level, figures can be represented as dictionaries and displayed using functions from the plotly.modeule. The fig dictionary in the example below describes a figure. 

## method 2 - Figure as Graph Objects
- the plotly.graph_objects module provides an automatically-generated hierarchy o classes called "graph objects" that may be used to represent figures, with a top-level class plotly.graph_objects.Figure.

## Recommended way - method 3 - plotly express
- the recommended way to create figures and populate them is to use potly express
- the plotlyexpress modle (usually iported as p) cotains functions that can create entire figures at once, and is referred to as plotly express or px
- plotly express is a built-in patrt of the lotly librry and is te recommended starting point for creating most common figures.
- every plotly express function uses graph objects internally and return a plotly graph_objects.figure instance
- plotly express provides more than 30 function for creating different types of figures.

## Method 3 - Plotly express -features
- Functions for creating graphs: scatter plot, line chart, pie chart, histgram, headmap, geoplot
- flexible inputs: can takes lists, dictionaries, ataframes
- automatic figure labelling
- automatic hover labels
- styling control

# CallBacks

- callbacks are used to cal python functions that are automatically called by dash whenever an input component's property changes.
- the inputs and output of  our application's interface are described declaratively as the arguments of the @app.calback decorator.
- By writing this decorator, we are telling dash to call this function for us whenever the value of the input component (the text box) changes in order to update the children of the output componen on the page (the html div)
- you can use any name for the function that is wrapped by the @app.callback decorator. The convenition is that the ame describes the callback.
- you can use an name for the functiona arguments, but you must use the same names inside te callbacl function as you do in its definition, just like in a regular python functio The arguments are positional: first the input items and then any state items are given in the same order as in te decorator.
- you must use the same id you gave a dash component in the app.layout when referring to it as either an input or output of the @app.callback decorator.
- the @app.callback decorator needs to be directly above the callback function declaration. If there is a blank line between the decorator and the function definition, the callback registraation will not be sucessful.
- in dash, the inputs and outputs of our application are siply the properties of a particular component. In his example, our input is te value property of the component taat has the id my input,. Our output is the chidren property of the component with the id my-output

- whenever a iput property changes the function tat the callback decorator wraps will get called automatically.
- Dash provides the function with new ew value of the input property as an input argument and dash updates the property of the output component with whatever was returned by the function

- notice how we do not set a value for the children property of the my-output component in the layout

- when the dash app starts, it automatically calls all o the callbacks with the initiial vlues of the input components oin order to populate the iniital state of the output components

- in this example, if you speified something like hmtl,,d iv(id="my-otput", chidren"hello world"), it woud get overwritten when the app starts

# Coding
- Update the PIP
```
python.exe -m pip install --upgrade pip
```

- Create gitignore file
```
python .\src\utils\gitignore_creater.py
```

- Create environment venv
```
python -m venv .venv
```

- Active environment venv
```
.venv\Scripts\Activate.ps1
```

- Install dependencies
```
pip install -r requirements.txt
```

- Validate dependencies
```
python -c "import requests, plotly, pandas, dash-bootstrap-components, dash"
```

# Directories

```
covid-dash-app/
├── app.py
├── cc3_cn_r.json
├── us_state_abbrev.json
├── covid_news_articles.csv
└── requirements.txt
```