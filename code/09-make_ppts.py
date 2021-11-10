# %%
from pptx import Presentation
import json
from os.path import join as ospj
import glob
from PIL import Image
import pandas as pd

def _add_image(slide, placeholder_id, image_url):
    placeholder = slide.placeholders[placeholder_id]

    # Calculate the image size of the image
    im = Image.open(image_url)
    width, height = im.size

    # Make sure the placeholder doesn't zoom in
    placeholder.height = height
    placeholder.width = width

    # Insert the picture
    placeholder = placeholder.insert_picture(image_url)

    # Calculate ratios and compare
    image_ratio = width / height
    placeholder_ratio = placeholder.width / placeholder.height
    ratio_difference = placeholder_ratio - image_ratio

    # Placeholder width too wide:
    if ratio_difference > 0:
        difference_on_each_side = ratio_difference / 2
        placeholder.crop_left = -difference_on_each_side
        placeholder.crop_right = -difference_on_each_side
    # Placeholder height too high
    else:
        difference_on_each_side = -ratio_difference / 2
        placeholder.crop_bottom = -difference_on_each_side
        placeholder.crop_top = -difference_on_each_side

# %%
# Get paths from config file and metadata
with open("config.json") as f:
    config = json.load(f)
repo_path = config['repositoryPath']
metadata_path = config['metadataPath']
palette = config['lightColors']

data_path = ospj(repo_path, 'data')
figure_path = ospj(repo_path, 'figures')

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort.xlsx"))

# %% Make subgraphs and expression ppt
prs = Presentation()
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]
    print(pt)

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    slide = prs.slides.add_slide(prs.slide_layouts[8])
    _add_image(slide, 1, ospj(pt_figure_path, 'subgraphs.png'))
    title_placeholder = slide.shapes.title
    title_placeholder.text = pt

    for fname in glob.glob(ospj(pt_figure_path, 'subgraph_expression_sz_*.png')):
        slide = prs.slides.add_slide(prs.slide_layouts[8])
        _add_image(slide, 1, fname)
        title_placeholder = slide.shapes.title
        title_placeholder.text = pt

    prs.save(ospj(figure_path, 'subgraphs_and_expression.pptx'))

# %%
