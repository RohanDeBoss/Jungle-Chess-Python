import os, sys, shutil
import numpy as np

SRC_SUFFIX, DST_SUFFIX = "_tb16.bin", "_tb8.bin"
CHUNK = 1 << 26


def convert(tb_dir):
    for f in sorted(os.listdir(tb_dir)):
        if not f.endswith(SRC_SUFFIX):
            continue
        src = os.path.join(tb_dir, f)
        dst = src[:-len(SRC_SUFFIX)] + DST_SUFFIX
        if os.path.exists(dst):
            print(f"skip   {os.path.basename(dst)}"); continue
        a = np.memmap(src, dtype=np.int16, mode='r')
        n = a.size
        tmp = dst + ".tmp"
        out = np.memmap(tmp, dtype=np.int8, mode='w+', shape=(n,))
        try:
            for s in range(0, n, CHUNK):
                blk = np.asarray(a[s:s + CHUNK])
                if blk.size and (int(blk.min()) < -127 or int(blk.max()) > 127):
                    raise OverflowError(f"{f}: |DTM| > 127")
                out[s:s + CHUNK] = blk.astype(np.int8)
            out.flush()
            for s in range(0, n, CHUNK):
                if not np.array_equal(np.asarray(a[s:s + CHUNK]).astype(np.int8),
                                      np.asarray(out[s:s + CHUNK])):
                    raise RuntimeError(f"{f}: verify mismatch")
        except Exception:
            del out; os.remove(tmp); raise
        del out
        os.replace(tmp, dst)
        if os.path.exists(src + ".wdl"):
            shutil.copyfile(src + ".wdl", dst + ".wdl")
        print(f"OK     {os.path.basename(dst)}  ({n:,} entries)")


if __name__ == "__main__":
    convert(sys.argv[1] if len(sys.argv) > 1 else
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "TBs"))
