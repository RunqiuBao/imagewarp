import numpy
import torch
from torch.autograd import Variable
import torch.nn as nn


def backwardwarp(x, flo):
    """
    Backwarp: given optical flow from img2 to img1, and img1, we can warp img1 to img2 with this function.

    x: numpy array. [B, C, H, W] (im1).
    flo: numpy array. [B, 2, H, W] (flow).

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, flo = x.to(device), flo.to(device)

    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(device)
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1] 
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
    mask = nn.functional.grid_sample(mask, vgrid)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    
    output = (output * mask).to('cpu').numpy()
    output = numpy.clip(output, 0, 255).astpye('uint8')
    return output


def weightFunc(x):
    return numpy.exp(-x)


def forwardwarp(img, flo, gain=1.0):
    """
    forward warping with splatting strategy. only consider splatting 4 corners around warp target.
    weight func is e^-x.
    Given optical flow from img1 to img2, and img1, we can warp img1 to img2 with this function.

    img: [B, C, H, W] (img1)
    flo: [B, 2, H, W] (flow)

    """
    B, C, H, W = flo.shape
    xx = numpy.tile(numpy.arange(0, W), (H, 1))
    yy = numpy.tile(numpy.arange(0, H)[:, numpy.newaxis], (1, W))
    grid = numpy.concatenate([xx[numpy.newaxis, ...], yy[numpy.newaxis, ...]], axis=0)
    grid = numpy.concatenate([grid[numpy.newaxis, ...] for i in range(B)], axis=0)
    vgrid = grid + flo
    batchSubResult = []
    for indexImage in range(B):
        ff_map = numpy.concatenate([numpy.floor(vgrid[indexImage][0])[..., numpy.newaxis], numpy.floor(vgrid[indexImage][1])[..., numpy.newaxis]], axis=-1)
        cf_map = numpy.concatenate([numpy.ceil(vgrid[indexImage][0])[..., numpy.newaxis], numpy.floor(vgrid[indexImage][1])[..., numpy.newaxis]], axis=-1)
        fc_map = numpy.concatenate([numpy.floor(vgrid[indexImage][0])[..., numpy.newaxis], numpy.ceil(vgrid[indexImage][1])[..., numpy.newaxis]], axis=-1)
        cc_map = numpy.concatenate([numpy.ceil(vgrid[indexImage][0])[..., numpy.newaxis], numpy.ceil(vgrid[indexImage][1])[..., numpy.newaxis]], axis=-1)
        allCornerMaps = [ff_map, cf_map, fc_map, cc_map]
        warpSubResults = []
        for cornerMap in allCornerMaps:
            cornerMap = cornerMap.reshape(-1, 2)
            intens = img[indexImage].transpose(1, 2, 0).reshape(-1, C)
            distances = weightFunc(numpy.linalg.norm(cornerMap - vgrid[indexImage].transpose(1, 2, 0).reshape(-1, 2), axis=1))
            maskNan = numpy.isnan(distances)  # Note: bothe cornerMap and 
            maskOutOfRangeX = numpy.logical_or(cornerMap[:, 0] >= W, cornerMap[:, 0] < 0)
            maskOutOfRangeY = numpy.logical_or(cornerMap[:, 1] >= H, cornerMap[:, 1] < 0)
            maskOuters = numpy.logical_or(maskNan, numpy.logical_or(maskOutOfRangeX, maskOutOfRangeY))
            cornerMap = cornerMap[~maskOuters].astype('int')
            intens = intens[~maskOuters]
            distances = distances[~maskOuters]
            weightedIntens = numpy.tile(distances[:, numpy.newaxis], (1, C)) * intens
            result = numpy.zeros((H, W, C), dtype='float')
            numpy.add.at(result, (cornerMap[:, 1], cornerMap[:, 0]), weightedIntens)
            warpSubResults.append(result)
        sumResults = numpy.zeros((H, W, C), dtype='float')
        for subResult in warpSubResults:
            sumResults += subResult
        sumResults = sumResults * gain
        sumResults = numpy.clip(sumResults, 0, 255).astype('uint8')
        batchSubResult.append(sumResults[numpy.newaxis, ...])

    return numpy.concatenate(batchSubResult, axis=0)