import sys, os, re
import subprocess as sp
import numpy as np
import timeit

from os.path import isfile, join

def parse_dir(dir_name):
  """ Get relevant info from directory name where the format
      is 8a4w-r0.4__stuff """
  m = re.match('ocs-(\d+)a(\d+)w-r([\d|.]+)', dir_name)
  if m:
    abits = int(m.group(1))
    wbits = int(m.group(2))
    r = float(m.group(3))
  else:
    #print('Cannot match directory %s' % dir_name)
    return None
  return abits, wbits, r

def dir_key(dir_name):
  """ Used for sorting """
  p = parse_dir(dir_name)
  if p is None:
      return 0
  abits, wbits, r = p
  return 1000*abits + 100*wbits + int(100*r)

def find_acc(fname):
  """ Find the Top1 and Top5 accuracy """
  with open(fname, 'r') as f:
    for l in f:
      m = re.search("==> Top1: ([\d|.]+)\W*Top5: ([\d|.]+)", l)
      if m:
        return float(m.group(1)), float(m.group(2))
    return None

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Give directory name")
    sys.exit(0)

  top_dir = sys.argv[1]

  # Find each experiment dir
  dirs = sorted( os.listdir(top_dir), key=dir_key, reverse=True )

  for d in dirs:
    # Get the bitwidths and ratio
    p = parse_dir(d)
    if p is None:
        continue
    abits, wbits, r = p

    # There should be exactly 1 logfile in each dir
    d = join(top_dir, d)
    logs = [f for f in os.listdir(d) if isfile(join(d,f)) and f[-4:]=='.log']
    assert(len(logs) == 1)
    logfile = join(d, logs[0])

    # Read the accuracy
    acc = find_acc(logfile)
    if acc:
      print("%d, %d, %4.2f, %6.3f, %6.3f" % (abits, wbits, r, acc[0], acc[1]))


