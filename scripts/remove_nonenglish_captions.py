# -*- coding: utf-8 -*-

import multiprocessing
import pickle
import re
import string
import sys

import langdetect
import regex
import torch
from tqdm import tqdm

try:
    _, CAPTION_FILE, IMG_FILE= sys.argv
except:
    print("Usage: {0} $CaptionPickleFile $ImageTorchFile".format(sys.argv[0]))
    exit(0)

START = "\\"
END = "\n"

###############################################################
###                Text Processing Utilities                ###
###############################################################

# https://stackoverflow.com/questions/26568722/remove-unicode-emoji-using-re-in-python
try:
    # Wide UCS-4 build
    EMOJI_RE = re.compile(u'['
                      u'\U0001F300-\U0001F64F'
                      u'\U0001F680-\U0001F6FF'
                      u'\u2600-\u26FF\u2700-\u27BF]+',
                      re.UNICODE)
except re.error:
    # Narrow UCS-2 build
    EMOJI_RE = re.compile(u'('
                      u'\ud83c[\udf00-\udfff]|'
                      u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
                      u'[\u2600-\u26FF\u2700-\u27BF])+',
                      re.UNICODE)

HASHTAG_RE = re.compile(r"[\#@]\w+", re.IGNORECASE)


def remove_emojis(line):
    return EMOJI_RE.sub("", line)


def simplify_letters(s):
    """ Returns a simplified version of a string s that essentially only
        contains Letters (according to Unicode). """
    permissable_chars = "•éíàćñäáûïâêåö"
    s = regex.sub(r'\p{Z}+', ' ', s)  # space characters
    s = regex.sub(r'\p{P}', "", s)  # punctuation chars
    s = regex.sub(r'\p{N}', "", s)  # numeric chars
    s = regex.sub(r'\p{S}', "", s)  # symbols
    s = regex.sub(r'\p{C}', "", s)  # control chars
    s = ''.join(c for c in s if c in string.printable)  # remove non-printable chars
    for pc in permissable_chars:
        s = s.replace(pc, "")
    return s


def clean_punctuation(l):
    """ Returns a simplified version of a string s that uses non-smart quotes
        and uniform-length hypens (from ASCII). Also removes non-printable
        characters. """
    l = l.replace('“', '"').replace('”', '"').replace("⠀", " ").replace("’", "'").replace("‘", "'").replace("…", "...")
    l = regex.sub(r'\p{Pd}', '-', l)  # fix hyphens
    l = l.replace(START, "").replace("\n", "\t")
    return l


def is_ASCII(s):
    """ Returns true if the string is essentially ASCII compatible (except
        for a few designated characters). """
    try:
        s = simplify_letters(s)
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError as e:
        return False
    except Exception as e:
        print(e, s)
        return False
    else:
        return True

def is_acceptable_english(line):
    """ Returns (line, True) if line is English. line will be modified to contain permissible characters and remove
        extraneous hyphens, etc."""
    line = clean_punctuation(line)
    try:
        line_wo_hashtags_emojis = remove_emojis(re.sub(HASHTAG_RE, "", line))
        lang = langdetect.detect(line_wo_hashtags_emojis)
        is_ascii = is_ASCII(line_wo_hashtags_emojis)
        is_eng = (lang == "en" and is_ascii)
    except:
        return (line, False)
    return (line, is_eng)


###############################################################
###                       Script Body                       ###
###############################################################

# IMAGES = torch.load(IMG_FILE)
# with open(CAPTION_FILE, "rb") as md:
#     CAPTIONS = pickle.load(md)
#
# def main_parallel():
#
#     p = multiprocessing.Pool(16)
#
#     caps_imgs = list(tqdm(p.imap(work, range(IMAGES.shape[0])), total=IMAGES.shape[0]))
#     good_captions, good_images = zip(*caps_imgs)
#     starting_len = len(good_captions)
#     good_images = [i for i in good_images if i is not None]
#     good_captions = [c for c in good_captions if c is not None]
#     ending_len = len(good_captions)
#     print(starting_len - ending_len, "captions removed.", ending_len, "left.")
#     new_img_file_path = IMG_FILE[:-4] + "_eng" + ".tch"
#     new_cap_file_path = CAPTION_FILE[:-4] + "_eng" + ".pkl"
#     with open(new_cap_file_path, "wb") as cf:
#         pickle.dump(good_captions, cf)
#     torch.save(torch.stack(good_images), new_img_file_path)
#
#
# def work(idx):
#     cap_md = CAPTIONS[idx]
#     try:
#         cap = cap_md["node"]["edge_media_to_caption"]["edges"][0]["node"]["text"]
#     except IndexError:
#         return None, None
#     cap, is_eng = is_acceptable_english(cap)
#     if is_eng:
#         return cap, IMAGES[idx]
#     else:
#         return None, None


def main():
    images = torch.load(IMG_FILE)
    good_images = []
    good_captions = []
    with open(CAPTION_FILE, "rb") as md:
        m = pickle.load(md)
        for cap_metadata, img_idx in tqdm(zip(m, range(images.shape[0])), total=images.shape[0]):
            try:
                cap = cap_metadata["node"]["edge_media_to_caption"]["edges"][0]["node"]["text"]
            except IndexError:
                continue
            cap, is_eng = is_acceptable_english(cap)
            if is_eng:
                good_captions.append(cap)
                good_images.append(images[img_idx])

    starting_len = images.shape[0]
    ending_len = len(good_captions)
    print(starting_len - ending_len, "captions removed.", ending_len, "left.")
    new_img_file_path = IMG_FILE[:-4] + "_eng.tch"
    new_cap_file_path = CAPTION_FILE[:-4] + "_eng.pkl"
    with open(new_cap_file_path, "wb") as cf:
        pickle.dump(good_captions, cf)
    torch.save(torch.stack(good_images), new_img_file_path)




if __name__ == "__main__":
    main()

