# cog-Point-E


[![Replicate](https://replicate.com/cjwbw/kpoint-earlo/badge)](https://replicate.com/cjwbw/point-e) 


![Animation of four 3D point clouds rotating](https://github.com/openai/point-e/blob/main/point_e/examples/paper_banner.gif?raw=true)

A Cog implementation for https://github.com/openai/point-e,
for [Point-E: A System for Generating 3D Point Clouds from Complex Prompts](https://arxiv.org/abs/2212.08751).

This implementation enables text2pointcloud (with [base_40m_textvec.pt](https://github.com/openai/point-e/blob/main/point_e/models/download.py#L16)) and img2pointcloud (with [base40M.pt](https://github.com/openai/point-e/blob/main/point_e/models/download.py#L18)) generation.


You can easily run with Replicate [API](https://replicate.com/cjwbw/point-e/api) or try the web [demo](https://replicate.com/cjwbw/point-e):


Two kinds of input are accepted: 
- a `prompt` for generating pointcloud from text, or 
- an `image` for generating pointcloud from the image
Note that if the `prompt` is provided, the `image` will be ignored. Therefore for effectively generating pointcloud from images please remove the `prompt` if it was previously set.

The supported return values are:
- a pointcloud file saved in `.npz` format
- [Optional] the samples of the points, which is converted from tensor to a list of size [1, 6, 4096] if `save_samples` is set to `True`. This may be more useful for API use cases
- [Optional]  a plt image with 9 veiws, if `generate_pc_plot` is set to `True`
- [Optional] we also enable generating an animation of the pointcloud if `generation_animation` is set to `True`.