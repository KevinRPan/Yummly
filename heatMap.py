import csv
from BeautifulSoup import BeautifulSoup

heat = {}
reader = csv.reader(open('countryProbabilities.csv'), delimeter=",")
for row in reader:
    try:
        heat[row[0].lower()] = float([row[1].strip())
    except:
        pass

svg = open('countries.svg', 'r').read()

soup = BeautifulSoup(svg, selfClosingTags=['defs', 'sodipodi:namedview', 'path'])

colors = ["#C46D8EF","9ECAE1","6BAED6","#2171B5","#0B4594"]

#Find countries with multiple polygons
gs = soup.contents[2].findAll('g',recursive=False)
#Find countries without multiple polygons
paths = soup.contents[2].findAll('path', recursive=False)

path_style = "fill"

for p in paths:
    if 'land' in p['class']:
        try:
            prob = heat[p['id']]
        except:
            continue
    if prob > 0.8:
        color_class = 5
    elif prob > 0.6:
        color_class = 4
    elif prob > 0.4:
        color_class = 3
    elif prob > 0.2:
        color_class = 2
    else:
        color_class = 1

    color = colors[color_class]
    p['style'] = path_style + color

for p in gs:
    if 'land' in p['class']:
        try:
            prob = heat[p['id']]
        except:
            continue
    if prob > 0.8:
        color_class = 5
    elif prob > 0.6:
        color_class = 4
    elif prob > 0.4:
        color_class = 3
    elif prob > 0.2:
        color_class = 2
    else:
        color_class = 1

    color = colors[color_class]
    p['style'] = path_style + color

    for t in p.findAll('path',recursive=True):
        t['style'] = path_style + color

f = open("heatMap.svg", "w")
f.write(str(soup).replace('viewbox','viewBox',1))
