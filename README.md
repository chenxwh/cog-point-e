# Point·E


This is a cog implementation of the official code from: https://github.com/openai/point-e.
See the paper [Point-E: A System for Generating 3D Point Clouds from Complex Prompts](https://arxiv.org/abs/2212.08751) for more details.

This repo includes text2pointcloud (with [base_40m_textvec.pt](https://github.com/openai/point-e/blob/main/point_e/models/download.py#L16) checkpoint) and img2pointcloud (with [base40M.pt](https://github.com/openai/point-e/blob/main/point_e/models/download.py#L18) checkpoint) generation.


## Web demo and API

Try the demo or explore the API here [![Replicate](https://replicate.com/cjwbw/kpoint-earlo/badge)](https://replicate.com/cjwbw/point-e) 


## Run locally

First, download the pre-trained weights:

    cog run script/download-weights 

Then, you can run predictions:

    cog predict -i prompt="a red motorcycle"


Two kinds of input are accepted: 

-  a `prompt` for generating point cloud from text, or 

-  an `image` for generating point cloud from the image
Note that if the `prompt` is provided, the `image` will be ignored. Therefore for effectively generating point cloud from images please remove the `prompt` if it was previously set.

The supported output format are:

-  [PointCloud](https://github.com/chenxwh/point-e/blob/main/point_e/util/point_cloud.py#L19) saved as `json_file`. `PointCloud` is  an array of points sampled on a surface, with `coords`: an [N x 3] array of point coordinates, and `channel` attributes which corresponds to `R`, `G`, `B` colors of the points in `coords`. We re-ordered the format to more standard way as follows:
    `{"coords": [...], "colors": [...]}`, 
where "coords" is an [N x 3] array of (X,Y,Z) point coordinates, and "colors" is an [N x 3] array of (R,G,B) color values

- Or an `animation` of the point cloud
