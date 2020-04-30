# HTML-Project

Requirements (recommend to use Conda, use _pip_ instead):
- python 3.7
- pyTorch\
`conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`
- beatutifulsoup4\
`conda install -c anaconda beautifulsoup4`
- numpy\
`conda install numpy`
- tqdm\
`conda install tqdm`
- matplotlib\
`conda install matplotlib`

1. to crawl the source code from Bootstrap web pages:\
`python crawler.py`\
It will shows the crawling url and the number of visited urls, unvisited urls and saved files like the following:\
GET https://getbootstrap.com/ 0 0 0\
GET https://getbootstrap.com//docs/4.4/getting-started/introduction/ 1 6 1\
GET https://getbootstrap.com//docs/4.4/components/pagination/ 2 71 2\
GET https://getbootstrap.com//docs/4.4/examples/ 3 71 3\
GET https://getbootstrap.com//docs/4.4/examples/sticky-footer/ 4 91 4\
GET https://getbootstrap.com//docs/4.4/content/reboot/ 5 90 5

2. to get all the class names from Bootstrap website: \
`python get_class.py`   
**warning**: the order of the class names may change every time
It will show a progression bar like:\
100%|██████████| 192/192 [00:05<00:00, 36.03it/s]\
and produce classes.txt file at root, containing all class names

3. Build the matrix counting frequencies of parentness between classes:\
`python matrix_model.py`\
It firstly crates two json files _name2idx.json_ and _idx2name.json_\
and then calculate the frequency matrix and save it to freq_matrix.npy
`numpy.load('freq_matrix.npy')` to see the actual values\
Finally it runs the model on test dataset and print out the Errors as the following:\
RelationError:		col-xl-2 and font-weight-bold has no relation\
ClassNameError:		no info-color class\
RelationError:		col-xl-2 and mb-4 has no relation

4. pre-process data:\
`python preprocess.py`\
It collects sequences of class parentness and save them to _train_ folder,
and it should produce 23829 npy files

5. the RNN model training:\
`python RNN_model.py train`\
It starts the training, printing out the progress like:\
Train Epoch: 1 [1920/23829 (8%)]        ACC: 14.08      Loss: 5.564291\
Train Epoch: 1 [2240/23829 (9%)]        ACC: 14.29      Loss: 5.624841\
Train Epoch: 1 [2560/23829 (11%)]       ACC: 14.15      Loss: 5.516018\
and save the model to _model_epoch_1/2/3...pt_

6. the RNN model testing:
`python RNN_model.py test`\
It evaluate the model on test dataset, if there is unseen class name, an error will be printed\
otherwise, it runs through the model and print out the probability of two classes to be parent and child.