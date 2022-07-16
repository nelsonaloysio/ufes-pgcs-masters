import collections
import io
import json
import math
import os
import re
import string
# import pickle
# import pkgutil
# import urllib
from functools import reduce
from itertools import chain
from typing import Callable, Union
# from pprint import pprint
# from urllib.request import urlopen

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import seaborn as sns
# import holoviews as hv
# import holoviews.operation.datashader as hd
from datashader.bundling import connect_edges, hammer_bundle
from plotly.subplots import make_subplots
from scipy.stats import norm
from sklearn.base import TransformerMixin
# from bokeh.io import show
# from bokeh.models import CategoricalColorMapper, ColumnDataSource
# from bokeh.palettes import RdBu3
# from bokeh.plotting import figure
# from datashader.layout import (circular_layout, forceatlas2_layout,
#                                random_layout)
# from holoviews.operation.datashader import datashade, dynspread
# from scipy.stats import hmean, rankdata
# from sklearn.linear_model import LogisticRegression
# from tldextract import extract as tld_extract

# https://github.com/nelsonaloysio/pygraphkit
from pygraphkit import GraphKit

# https://gist.github.com/nelsonaloysio/302dbbf3963fababde6e9f97669587df
from stopwords import CUSTOM_STOPWORDS

ACCENT_REPLACEMENTS = {
    ord('á'): 'a', ord('ã'): 'a', ord('â'): 'a',
    ord('à'): 'a', ord('è'): 'e', ord('ê'): 'e',
    ord('é'): 'e', ord('í'): 'i', ord('ì'): 'i',
    ord('ñ'): 'n', ord('ò'): 'o', ord('ó'): 'o',
    ord('ô'): 'o', ord('õ'): 'o', ord('ù'): 'u',
    ord('ú'): 'u', ord('ü'): 'u', ord('ç'): 'c'}

VALID_CHARACTERS = "@#"
INVALID_CHARACTERS = "\\\"'’…|–—“”‘„•¿¡"

CHARACTER_REPLACEMENTS = str.maketrans('', '', ''.join(
    set(string.punctuation + INVALID_CHARACTERS) - set(VALID_CHARACTERS)))

IGNORE_STARTS_WITH = ['http', 'www', 'kkk']

DEFAULT_TEMPLATE = "none"
FONT_FAMILY = 'Raleway, Arial, sans-serif'
FONT_COLOR = 'grey'
LEGEND_YREF = 'paper'
TEXT_POSITION = 'top center'

FONT_SIZE = 16
LEGEND_Y = 0.5
MARKER_SIZE = 6

AUTORANGE = True
CONNECT_GAPS = False

SUBPLOT_HEIGHT = 768

COLORS = [
    '#006cb7', '#ff7700',
    '#00b035', '#ed0000',
    '#a643bd', '#965146',
    '#fb4cbe', '#7f7f7f',
    '#b2cb10', '#00c2d3']

COLORSCALE = [
    [0,    "rgb(178, 34, 34)"],
    [0.25, "rgb(255, 140, 0)"],
    [0.5,  "rgb(255, 255, 51)"],
    [0.75, "rgb(40, 60, 190)"],
    [1,    "rgb(230, 230, 250)"],
]

COLORS_COMMUNITIES = {
    0: "#006cb7",
    1: "#ff7700",
    2: "#00b035",
    3: "#ed0000",
    4: "#a643bd",
    5: "#965146",
    6: "#eec250",
}

DS_LAYOUTS = ['forceatlas2',
              'circular',
              'random']

DS_OPTS = dict(plot_height=500, plot_width=500)
HV_OPTS = dict(width=400, height=400, xaxis=None, yaxis=None, padding=0.1)

plt.rc("savefig", dpi=600)
sns.set(font_scale=.75)

class Tokenizer(TransformerMixin):

    def __init__(self, max_paragraphs=None, stop_words=CUSTOM_STOPWORDS, **kwargs):
        self.max_paragraphs = max_paragraphs
        self.stop_words = stop_words

    def fit(self, X, y=None):
        ''' Just returns class, nothing to fit. '''
        return self

    def transform(self, X):
        ''' Abstract method for "DIY" transformations. '''
        return [
            ' '.join(
                ' '.join(
                    self.tokenize(sent)
                )\
                for sent in (
                    x.split('\n')[:self.max_paragraphs]
                    if isinstance(x, str) else ''
                )
            )
            for x in X
        ]

    def tokenize(self, sentence: str):
        '''
        Returns word token, cleared from emojis, accents and punctuation.
        '''
        return [
            x
            .replace('](', ' ')
            .translate(ACCENT_REPLACEMENTS)
            .translate(CHARACTER_REPLACEMENTS)
            for x in
                self.clear_emojis(sentence)
                .lower()
                .split()
            if
                len(x) > 2
            and
                x.strip(VALID_CHARACTERS) not in self.stop_words
            and
                not self.is_number(x)
            and
                not any(x.startswith(_) for _ in IGNORE_STARTS_WITH)
        ]

    @staticmethod
    def is_number(str_word):
        '''
        Check string as an integer or float.
        '''
        try:
            int(str_word)
        except:
            try:
                float(str_word)
            except:
                return False
        return True

    @staticmethod
    def clear_emojis(str_text, replace_with=r' '):
        '''
        Returns string after clearing from emojis.
        '''
        return re\
            .compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"  # extra (1)
                u"\U000024C2-\U0001F251"  # extra (2)
                u"\U0000200B-\U0000200D"  # zero width
                "]+", flags=re.UNICODE)\
            .sub(replace_with, str_text)

    @staticmethod
    def ngrams(tokens: list, n=2):
        '''
        Returns n-grams from list of tokens.
        '''
        return [
            g
            for g in
                list(nltk.ngrams(tokens, n))
            if
                len(set(g)) == n
        ]


class DataShaderPlot():

    def __init__(self):
        """ Initializes class. """

    def ds_plot(self, G, pos={}, layout='forceatlas2', method='connect', name='',
        bw=0.05, cat=None, inline=False, output=None, kwargs=DS_OPTS):
        '''
        Returns graph plot using DataShader algorithms.
        Defaults to random position layout if none given.

        Input parameters:
        * G: network graph object
        * pos: node positions to plot
        * layout: from ds_layout()
        * method: connect; bundle
        * name: title or label
        * bw: initial bandwith for bundled
        * cat: node category in graph
        * output: write image to file
        * kwargs: default parameters
        '''
        nodes = pd.DataFrame([x for x in G.nodes], columns=['id'])
        nodes.set_index('id', inplace=True)
        try: data = [[x, *pos.loc[x]] for x in G.nodes]
        except: data = [[x, *pos[x]] for x in G.nodes]
        columns = ['id', 'x', 'y']
        nodes = pd.DataFrame(data, columns=columns)
        if cat: # store category
            columns.append(cat)
            nodes[cat] = pd.Series([G._node[x][cat] for x in G.nodes]).astype('category')
        nodes.set_index('id', inplace=True)
        edges = pd.DataFrame(list(G.edges), columns=['source', 'target'])
        if method == 'bundle':
            edges = hammer_bundle(nodes, edges, initial_bandwidth=bw)
        else: # lightweight
            edges = connect_edges(nodes, edges)
        graph = self.graph_plot(nodes, edges, name, cat=cat, kwargs=kwargs)
        if output != None:
            ds.utils.export_image(img=graph, filename=output, fmt='.png', background='white')
        if inline:
            return self.tf_plot([graph])
        return graph

    def nodes_plot(self, nodes, name=None, canvas=None, cat=None, kwargs=DS_OPTS, color="#cccccc"):
        '''
        Plot nodes using datashader Canvas functions.
        Returns datashader.transfer_functions.spread().
        '''
        canvas = ds.Canvas(**kwargs) if canvas is None else canvas
        aggregator = None if cat is None else ds.count_cat(cat)
        agg = canvas.points(nodes,'x','y',aggregator)
        return tf.spread(tf.shade(agg, alpha=180, min_alpha=180, color_key=COLORS_COMMUNITIES), px=5, name=name) # 150/150/3 #  cmap=color

    def edges_plot(self, edges, name=None, canvas=None, kwargs=DS_OPTS):
        '''
        Plot edges using datashader Canvas functions.
        Returns datashader.transfer_functions.shade().
        '''
        canvas = ds.Canvas(**kwargs) if canvas is None else canvas
        return tf.shade(canvas.line(edges, 'x','y', agg=ds.count()), name=name, cmap=["gray"], alpha=100) # 125

    def graph_plot(self, nodes, edges, name="", canvas=None, cat=None, kwargs=DS_OPTS, color="#cccccc"):
        '''
        Plot graph using datashader Canvas functions.
        Returns datashader.transfer_functions.stack().
        '''
        if canvas is None:
            xr = nodes.x.min(), nodes.x.max()
            yr = nodes.y.min(), nodes.y.max()
            canvas = ds.Canvas(**kwargs, x_range=xr, y_range=yr)
        np = self.nodes_plot(nodes, name + " nodes", canvas, cat, color=color)
        ep = self.edges_plot(edges, name + " edges", canvas)
        return tf.stack(np, ep, how="over", name=name)

    def tf_plot(self, figs, cols=1):
        '''
        Returns figures to plot split by the number of
        columns set using datashader.transfer_functions().

        Useful for displaying multiple images (IPython).

        Input parameters:
        * figs: list of figures or graphs
        * cols: number of columns to plot
        '''
        plots = []
        plots.append(fig for fig in figs)
        return tf.Images(*chain.from_iterable(plots)).cols(cols)


class Plot():

    def __init__(self, template=DEFAULT_TEMPLATE):
        ''' Initializes class with default template. '''
        pio.templates.default = template

    @staticmethod
    def plot(
        data: Union[dict, list, pd.DataFrame, pd.Series],
        x: Union[str, int, None] = None,
        y: Union[str, int, None] = None,
        graph: str = 'scatter',
        layout: str = 'layout',
        layout_opts: dict = {},
        name: dict = {},
        size: dict = {},
        text: Union[dict, list] = None,
        resizer: Callable[[float], float] = lambda x: x,
        **opts,
    )-> go.Figure:
        '''
        Returns a Plotly graph object figure from a dictionary,
        a list of categories or a Pandas data frame or series.
        '''
        layout = getattr(Plot, layout)(**layout_opts)

        if not y and isinstance(data, pd.DataFrame):
            raise RuntimeError(
                "Missing required 'y' attribute for building Plotly figures from Pandas.DataFrame objects.")

        return go.Figure(
            data=[
                getattr(Plot, graph)(
                    x=list(trace.keys())
                    if isinstance(trace, dict)
                    else trace[x].values
                    if x and isinstance(trace, pd.DataFrame)
                    else trace.index,
                    y=list(trace.values())
                    if isinstance(trace, dict)
                    else trace[y].values
                    if isinstance(trace, pd.DataFrame)
                    else trace.values,
                    name=name.get(index, index),
                    size=resizer(size.get(index, MARKER_SIZE)),
                    text=text.get(index, '') if isinstance(text, dict) else text,
                    **opts,
                )
                for index, trace in (
                    data.items()
                    if len(data)
                    and isinstance(data, dict)
                    and type(list(data.values())[0])
                    in (dict, pd.DataFrame, pd.Series)
                    else enumerate(data)
                    if isinstance(data, list)
                    else [(None, data)]
                )
            ],
            layout=layout,
        )

    @staticmethod
    def subplots(
        data: dict,
        graph: str = 'scatter',
        orient: str = 'ver',
        layout: str = 'layout',
        layout_opts: dict = {},
        rows: int = None,
        cols: int = None,
        title: str = None,
        **opts,
    ) -> go.Figure:
        ''' Returns a Plotly figure with subplots. '''
        cursor = [0, 0]

        if orient not in ('ver', 'hor'):
            return ValueError(
                f"Received invalid orient parameter: {orient}. Available choices: ('hor', 'ver')."
            )
        pointer = (0 if orient == 'ver' else 1)
        cursor[(1 if orient == 'ver' else 0)] += 1

        if cols is None:
            cols = (len(data)/(rows or len(data))) or 1
            cols = int(cols) + (1 if float(cols) != int(cols) else 0)

        if rows is None:
            rows = (len(data)/cols) or 1
            rows = int(rows) + (1 if float(rows) != int(rows) else 0)
        limit = (rows if orient == 'ver' else cols)

        fig = make_subplots(rows=rows, cols=cols)
        for key, trace in reversed(data.items()):
            cursor[pointer] += 1
            fig.append_trace(
                getattr(Plot, graph)(
                    list(trace.keys()),
                    list(trace.values()),
                    name=key,
                    **opts,
                ),
                row=cursor[0],
                col=cursor[1],
            )
            if cursor[pointer] == limit:
                cursor[pointer] = 0
                cursor[pointer-1] += 1

        fig.update_layout({'height': SUBPLOT_HEIGHT, **layout_opts})
        return fig

    @staticmethod
    def bar(x, y, **opts):
        '''
        Returns Plotly 2-dimensional scatter.

        Input parameters:
            * x: list of values for horizontal axis
            * y: list of values for vertical axis
            * name: to include in point information
            * text: to include in trace information
        '''
        return go.Bar(
            x=x,
            y=y,
            name=opts.get('name', ''),
            text=opts.get('text', ''),
            textfont=dict(
                family=FONT_FAMILY
            ),
        )

    @staticmethod
    def choropleth(x, y, **opts):
        '''
        Returns Plotly 2-dimensional choropleth.

        Input parameters:
            * x: country codes
            * y: absolute values
            * opts: optional parameters
            * text: name from contries
            * title: name of figure
        '''
        return go.Choropleth(
            locations=x,
            z=y,
            text=opts.get('text', []),
            colorscale=COLORSCALE,
            autocolorscale=False,
            reversescale=True,
            marker=go.choropleth.Marker(
                line=go.choropleth.marker.Line(
                    color='rgb(180,180,180)',
                    width=0.5,
                ),
            ),
            colorbar=go.choropleth.ColorBar(
                tickprefix='',
                title=opts.get('title', ''),
            ),
        )

    @staticmethod
    def geolayout(projection='equirectangular', text='', title=''):
        '''
        Returns Plotly.Figure() layout for choropleth maps.

        Input parameters:
            * title: Plotly figure title
            * x_title: horizontal axis title
            * y_title: vertical axis title
        '''
        return go.Layout(
            title=go.layout.Title(
                text=title,
            ),
            geo=go.layout.Geo(
                showframe=False,
                showcoastlines=False,
                projection=go.layout.geo.Projection(
                    type=projection,
                ),
            ),
            annotations=[go.layout.Annotation(
                x=0.55,
                y=0.1,
                #xref='paper',
                #yref='paper',
                text=text,
                showarrow=False)
            ],
        )

    @staticmethod
    def layout(title='', x_title='', y_title='', **opts):
        '''
        Returns Plotly.Figure() layout dictionary.

        Input parameters:
            * title: Plotly figure title
            * x_title: horizontal axis title
            * y_title: vertical axis title
        '''
        return go.Layout(
            xaxis=dict(
                autorange=AUTORANGE,
                title=x_title,
                ),
            yaxis=dict(
                autorange=AUTORANGE,
                title=y_title,
                ),
            legend=dict(
                y=LEGEND_Y,
                font=dict(
                    family=FONT_FAMILY,
                    size=FONT_SIZE,
                    color=FONT_COLOR,
                    ),
                ),
            title=title,
            **opts,
        )

    @staticmethod
    def scatter(x, y, mode='lines+markers', size=None, name='', text='', **opts):
        '''
        Returns Plotly 2-dimensional scatter.

        Input parameters:
            * x: list of values for horizontal axis
            * y: list of values for vertical axis
            * mode: 'lines', 'markers' or both (default)
            * name: to include in point information
            * text: to include in trace information
        '''
        return go.Scatter(
            x=x,
            y=y,
            name=name,
            text=text,
            mode=mode,
            connectgaps=CONNECT_GAPS,
            textposition=TEXT_POSITION,
            textfont=dict(
                family=FONT_FAMILY,
                ),
            marker=dict(
                size=size if size is not None else MARKER_SIZE,
                ),
            **opts,
            )

    @staticmethod
    def colors(colors=COLORS):
        '''
        Returns a sequence generator of discrete colors.

        See references below for built-in Plotly sequences:
            https://plotly.com/python/builtin-colorscales/
            https://plotly.com/python/colorscales/
            https://plotly.com/python/templates/
        '''
        while 1:
            for clr in colors:
                yield clr


def flatten(lst, n=None):
    return [x[:n] if n else x for x in lst for x in x]


def load_json(f, json_key=None):
    method = pd.Series if json_key else pd.json_normalize
    with open(f, "r") as j:
        return method([json.loads(x)[json_key] if json_key else json.loads(x) for x in j.readlines()])


def heatmap(df, annot=False, cmap="RdYlBu_r", **kwargs):
    return sns.heatmap(df, annot=annot, cmap=cmap, **kwargs)


def ph(f):
    output_name = "{}.csv".format(os.path.splitext(os.path.basename(f))[0])
    print(f"{f} => {output_name}")
    with open(f, "r") as j:
        with open(output_name, "w") as output_file:
            output_file.write("id,source,target,times\n")
            while True:
                line = j.readline()
                if line:
                    obj = json.loads(line)
                    output_file.write(f'{obj["id_str"]},{obj["user"]["screen_name"]},{obj["retweeted_status"]["user"]["screen_name"]},{obj["created_at"]}\n')


def graph_histogram(G, title="Histograma de grau", color='#000000'):
    deg = sorted([d for n, d in G.degree()], reverse=True)
    degree_count = collections.Counter(deg)
    deg, cnt = zip(*degree_count.items())
    fig, ax = plt.subplots(figsize=(3, 2.5),)
    plt.bar(deg, cnt, width=0.80, color=color)
    # plt.title(title, fontsize=10, loc='center')
    plt.ylabel("Número de nós", fontsize=12)
    plt.xlabel("Número de conexões (grau)", fontsize=12)
    ax.set_xticks([d for d in deg])
    ax.set_xticklabels(d for d in deg)#, fontsize=12)
    plt.plot()


def gen_user_corr(dct, **kwargs):
    d1 = pd.DataFrame({
        key: {
            k: df.index.intersection(dct[k].index).shape[0] for k in dct}
        for key, df in dct.items()
        })
    d2 = d1.divide(d1.max(), axis=0)

    fig, ax = plt.subplots(figsize=(6,5))         # Sample figsize in inches
    heatmap(d2, annot=True, center=.5,  linewidths=0, ax=ax, **kwargs)
    plt.show()

    return d1.astype(str).applymap(lambda x: x) + d2.multiply(100).round(2).applymap(lambda x: f' ({x}%)'.replace('.0%','%').replace(".",",")).astype(str)


def describe(df, columns=[]):
    '''
    Return most common statistics in a data frame.
    Accepts a list of columns to consider (optional).

    Input parameters:
        * df: Pandas data frame
        * columns: list of columns
    '''
    dd = df.describe()

    if not columns:
        columns = dd.columns

    elif isinstance(columns, str):
        columns = columns.split(',')

    for c in columns:
        mean = df[c].mean()
        median = df[c].median()
        mad = df[c].mad() # mean absolute deviation
        std = df[c].std() # sample standard deviation (S)
        var = df[c].var() # unbiased variance (S²)
        cova = var/mean # coefficient of variation
        #dd.loc['mean'], c] = mean
        #dd.loc['median', c] = median
        #dd.loc['std', c] = std # standard deviation
        dd.loc['mad', c] = mad # mean absolute deviation
        dd.loc['var', c] = var # variance index
        dd.loc['cova', c] = cova # coefficient of variane
        dd.loc['10%', c] = df[c].quantile(0.1) # 10% percentile
        #dd.loc['25%', c] = df[c].quantile(0.25) # first quartile
        #dd.loc['50%', c] = df[c].quantile(0.5) # same as median
        ##dd.loc['75%', c] = df[c].quantile(0.75) # last quartile
        dd.loc['90%', c] = df[c].quantile(0.9) # 90% percentile

    cols = ['count', 'mean', 'mad', 'std', 'min',
            '10%', '25%', '50%', '75%', '90%', 'max',
            'var', 'cova']

    return dd.reindex(cols)


def nx_plot(
    G,
    pos={},
    group='partition',
    colors=['r,b,c,m,y,g'],
    directed=None,
    edge_alpha=0.9,
    edge_color='gray',
    edge_width=0.15,
    figsize=(8,8),
    font_size=12,
    layout='kamada_kawai_layout',
    layout_opts={},
    node_shape='8',
    node_size=30,
    node_sizes=[],
    output=None,
    set_labels=[],
    show_labels=False,
    title='',
):
    node_cat = []
    node_colors = []
    directed = G.is_directed() if directed is None else directed

    for node in G.nodes(data=True):
        if group in node[1]:
            if node[1][group] not in node_cat:
                node_cat.append(n[1][cat])
            node_colors.append(colors[node_cat.index(n[1][cat]) % len(colors)])
            raise RuntimeError('Error: all nodes must have a category.')

    node_color = colors

    fig = plt.figure(1, figsize=figsize, dpi=80, facecolor='w', edgecolor='g')
    ax = fig.add_subplot(1,1,1)
    ax = plt.subplot(111)
    ax.set_title(title, fontsize=12, loc='left', y=1)

    if node_sizes:
        max_size = max(node_sizes)
        node_size = [10+int((x/max_size)*node_size) for x in node_sizes]

    if not pos:
        if layout is not None:
            pos = getattr(nx, layout)(G, **layout_opts)
        else:
            k = (4/math.sqrt(G.order())) if not k else k
            pos = nx.spring_layout(G, k=k)

    nx.draw_networkx_nodes(G, pos=pos, node_color=node_color, node_shape=node_shape, node_size=node_size) # ax=ax
    nx.draw_networkx_edges(G, pos=pos, edge_color=edge_color, width=edge_width, alpha=edge_alpha, arrows=directed) # ax=ax

    if show_labels:
        L = G
        if set_labels:
            L = nx.subgraph(L, set_labels)
        nx.draw_networkx_labels(L, pos=pos, font_size=font_size)#, ax=ax)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output, format='PNG') if output else None


def plotstats(d, pos, k, ylim=None, fmtr=lambda x:x, legend=False,):
    ax = plt.subplot(pos)
    plt.plot(d[k].index, d[k].values, marker='.')
    plt.title(str(k))
    ax.yaxis.set_major_formatter(fmtr)
    # xlim = ax.get_xticks()
    # ax.set_xlim([xlim[0], xlim[-1]])
    # if ylim:
    #     ax.set_ylim(ylim)
    # if legend:
    #     plt.legend(title_fontsize=12, bbox_to_anchor=(.5, -.25),
    #             loc='lower center', fontsize=12, ncol=5, columnspacing=1)
    # plt.grid(color='gray', linestyle='--', linewidth=0.15)


def plots(d, y=None, key=None, kind="barh", n=15):
    for y in (y if type(y) == list else [y] if type(y) == str else sorted(d.keys())):
        fig = plt.figure()
        ax = (d[y][key] if key else d[y]).value_counts(ascending=False)[:n].sort_values().plot(kind=kind)
        ax.set_title(f"{y}")
        ax.plot()

def is_clean(t: str, list_of_stopwords=[]):
    '''
    Returns word token, cleared from emojis, accents and punctuation.
    '''
    for x in t.split(" "):
        x = x\
            .replace('](', ' ')\
            .translate(ACCENT_REPLACEMENTS)\
            .translate(CHARACTER_REPLACEMENTS)
        for _ in INVALID_CHARACTERS:
            x = x.replace(_, "")
        if len(x) < 2 \
            or is_emoji(x) \
            or is_number(x.strip(VALID_CHARACTERS)) \
            or any(_ in x for _ in IGNORE_STARTS_WITH)\
            or is_stopword(x, list_of_stopwords):
                return False
    return True

def is_number(str_word):
    '''
    Check string as an integer or float.
    '''
    try:
        int(str_word)
    except:
        try:
            float(str_word)
        except:
            return False
    return True

def is_emoji(str_text, replace_with=r''):
    '''
    Returns string after clearing from emojis.
    '''
    x = re\
        .compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"  # extra (1)
            u"\U000024C2-\U0001F251"  # extra (2)
            u"\U0000200B-\U0000200D"  # zero width
            "]+", flags=re.UNICODE)\
        .sub(replace_with, str_text)
    return True if x == "" else False

def is_stopword(x, list_of_stopwords=[]):
    if x in list_of_stopwords:
        return True
    return False

def normdf(x):
    return norm.df(x, x.mean(), x.std())

def scale(ar):
    return (ar - ar.min()) / (ar.max() - ar.min())

def zero_centered_scale(ar):
    scores = np.zeros(len(ar))
    scores[ar > 0] = scale(ar[ar > 0])
    scores[ar < 0] = -scale(-ar[ar < 0])
    return (scores + 1) / 2.

def get_features(
    series: pd.Series,
    groupby: pd.Series = None,
    ignore_startswith: list = ["@","#"],
    n_grams: int = 1,
    normalized: bool = False,
    tfidf: bool = False,
    top_features: bool = False,
    map_func = lambda x: x,
) -> dict:

    if not series.shape[0]:
        return dict()

    groups = series\
        .groupby(groupby if groupby is not None else [0] * series.shape[0])\
        .apply(lambda x: list(set(x.index.tolist())))

    features = {
        group:
            series
            .loc[groups[group]]
            .astype(str)
            .apply(lambda x: [x for x in x.split() if not any(x.startswith(char) for char in ignore_startswith)])
            .apply(lambda x: x if n_grams == 1 else Tokenizer.ngrams(x, n_grams))
            .apply(lambda x: list(set(x)) if tfidf else x)
            .explode()
            .dropna()
            .value_counts()
        for group in groups.index
    }

    if tfidf: # tf(t,d)/log(N/df(t))
        N = series.shape[0]
        df = reduce(lambda x, y: x.add(y, fill_value=0), features.values())
        features = {
            group:
                (tf/tf.max())
                .divide(
                    (df.loc[tf.index]/df.loc[tf.index].max())
                    .apply(lambda x: math.log10(N/(x+1)))
                )
                .sort_values(ascending=False)
            for group, tf in features.items()
        }

    if normalized:
        features = {
            group:
                series.apply(lambda x: x/x.max(), axis=0)
            for group, series in features.items()
        }

    if top_features:
        features = pd\
            .Series({
                group:
                    series.index[:(len(groups)+1)]
                for group, series in features.items()})\
            .explode()\
            .drop_duplicates(keep='first')
        features = features\
            [~features.index.duplicated()]\
            .dropna()\
            .map(map_func)\
            .to_dict()

    return features if groupby is not None else features[0]
