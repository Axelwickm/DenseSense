from argparse import ArgumentParser
from pycocotools.coco import COCO
import os
import urllib
import workerpool


parser = ArgumentParser()
parser.add_argument("dataset", help="Which coco dataset to download",
                    choices=["val2014", "train2014", "val2017", "train2017"], type=str)


def main():
    args = parser.parse_args()
    dataset = args.dataset

    # Load annotations
    annFile = '../annotations/instances_{}.json'.format(dataset)
    assert os.path.isfile(annFile)
    cocoPath = './{}'.format(dataset)
    try:
        os.mkdir(cocoPath)
    except FileExistsError:
        pass

    # Init coco
    coco = COCO(annFile)
    personCatID = coco.getCatIds(catNms=['person'])[0]
    cocoImageIds = coco.getImgIds(catIds=personCatID)

    print("Putting all urls into big list!")
    urls = []
    for i in cocoImageIds:
        cocoImg = coco.loadImgs(i)[0]
        urls.append(cocoImg["coco_url"])

    saveto = [os.path.join(cocoPath, os.path.basename(url))
              for url in urls]

    print("Starting download of {} items".format(len(urls)))
    # Thank you to: https://github.com/shazow/workerpool/wiki/Mass-Downloader
    pool = workerpool.WorkerPool(size=6)

    # Perform the mapping
    pool.map(urllib.request.urlretrieve, urls, saveto)

    # Send shutdown jobs to all threads, and wait until all the jobs have been completed
    pool.shutdown()
    pool.wait()

    print(cocoImg)


if __name__ == '__main__':
    main()