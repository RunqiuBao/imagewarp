#!/usr/bin/python3
import argparse
from pathlib import Path

import imagewarp


def main():
    parser = argparse.ArgumentParser(description = "A commandline tool for warping and interpolating frames using forward or backward optical flow.\n"
    + "Definition of forward and backward warp:\n"
    + "- Forward warp: given known frame img1 and optical flow from img1 to img2, warp img1 to img2.\n"
    + "- Backward warp: given know frame img1 and optical flow from img2 to img1, warp img1 to img2.\n")
    parser.add_argument("-i", "--imgFilePath", required=True, type=str, help="Path to the input images.")
    parser.add_argument("--forwardFlowFilePath", default=None, help="Path to the forward flow files. For now flow file needs to be virtual-kitti format.")
    parser.add_argument("--backwardFlowFilePath", default=None, help="Path to the forward flow files. For now flow file needs to be virtual-kitti format.")
    parser.add_argument("-o", "--output_path", help="Path to save the output data.")
    parser.add_argument("--upsampleFrameRatio", required=True, help="interpolate into a ?x frame seqs. (x - 1) frames to interpolate between every 2 continuous source imgs.")
    parser.add_argument('--src_file_stye', type=str, default="%06d.jpg", help='img file name style')
    parser.add_argument('--dst_file_stye', type=str, default="%06d.png", help='output img filename style')
    args = parser.parse_args()

    assert (not (args.forwardFlowFilePath is None and args.backwardFlowFilePath is None)), "Args Error: no flow files available."

    imagewarp.eventshow(
        args.rw_module,
        Path(args.input_file),
        Path(args.output_path),
        dt_ms=int(args.dtms) if args.dtms else None,
        numevents_perslice=int(args.numevents) if args.numevents else None,
        is_use_concentrate=args.concentrate if args.concentrate else False,
        num_frames_exit=int(args.numframes) if args.numframes else None,
        is_save_lmdb=args.savelmdb if args.savelmdb else False,
    )


if __name__ == "__main__":
    main()
