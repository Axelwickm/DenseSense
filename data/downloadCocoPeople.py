from argparse import ArgumentParser
from pycocotools.coco import COCO
import os
from parfive import Downloader

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
        annIds = coco.getAnnIds(imgIds=cocoImg["id"], catIds=personCatID, iscrowd=None)
        annotation = coco.loadAnns(annIds)[0]
        if annotation["iscrowd"] == 0:
            urls.append(cocoImg["coco_url"])

    print("Enqueueing download of {} items".format(len(urls)))
    dl = Downloader()

    for url in urls:
        dl.enqueue_file(url, path=cocoPath)

    print("Downloading files...")
    dl.download()


if __name__ == '__main__':
    main()