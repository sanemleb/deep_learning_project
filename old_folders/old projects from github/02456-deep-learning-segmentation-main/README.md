# 02456-deep-learning
This is the project in the course: 02456 Deep Learning at DTU compute. The project case is given by Deloitte and is within segmentation of car parts on images of cars. The notebook showing all the results (recreating_results_notebook.ipynb) is found in the root of the repository. 

An overview of the structure of the project can be seen below:
- ./recreating_results_notebook.ipynb
- ./smp
  - train.py   
- ./unet
  - train.py
  - unet.py
- ./utils
  - /background_rem
    - detectron_background_removal.ipynb
    - bg_manager.py
    - bg_remover.py  
  - /dataloader
    - car_dataset.py   
  - /data_prep
    - cycle_gan_files.txt
    - test_data.txt
    - data_filter.py
    - train_test_val.py   
  - /visualization  
    - performance_graph.py
    - random_img_viz.py
    - test_viz.py   
