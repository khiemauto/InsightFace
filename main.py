from core import insight_face
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-fp", "--folders_path", default="dataset/photos",
                help="path to save folders with images")
ap.add_argument("-dbp", "--db_folder_path", default="database",
                help="path to save database")
ap.add_argument("-rdb", "--reload_db", type=int, default=0,
                help="reload database")
args = vars(ap.parse_args())

insight_face.main(args)