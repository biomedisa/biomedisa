from biomedisa.features.biomedisa_helper import *
from tifffile import imread
import numpy as np
import time
import numba

@numba.jit(nopython=True)
def reduce_blocksize(data, value=1, buff=64):
    zsh, ysh, xsh = data.shape
    argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = zsh, 0, ysh, 0, xsh, 0
    for k in range(zsh):
        for l in range(ysh):
            for m in range(xsh):
                if data[k,l,m]==value:
                    argmin_x = min(argmin_x, m)
                    argmax_x = max(argmax_x, m)
                    argmin_y = min(argmin_y, l)
                    argmax_y = max(argmax_y, l)
                    argmin_z = min(argmin_z, k)
                    argmax_z = max(argmax_z, k)
    argmin_x = max(argmin_x - buff, 0)
    argmax_x = min(argmax_x + buff, xsh)
    argmin_y = max(argmin_y - buff, 0)
    argmax_y = min(argmax_y + buff, ysh)
    argmin_z = max(argmin_z - buff, 0)
    argmax_z = min(argmax_z + buff, zsh)
    return argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x

def get_boundaries(slc):
    grad = np.zeros(slc.shape, dtype=np.uint8)
    tmp = np.abs(slc[:-1] - slc[1:])
    tmp[tmp>0]=1
    grad[:-1] += tmp
    grad[1:] += tmp
    tmp = np.abs(slc[:,:-1] - slc[:,1:])
    tmp[tmp>0]=1
    grad[:,:-1] += tmp
    grad[:,1:] += tmp
    grad[grad>0]=1
    return grad

def get_block_bounds(z_dim, block_size=1024, overlap=192):
    """
    Iterate over z-blocks with overlap. If the last block would be smaller than half block_size,
    it is merged into the previous block.
    """
    step = block_size - overlap
    block_bounds = []

    z_start = 0
    while z_start + block_size < z_dim:
        z_end = z_start + block_size
        block_bounds.append((z_start, z_end))
        z_start += step

    # Handle the last block
    if not block_bounds:
        # Entire volume is smaller than one block
        block_bounds.append((0, z_dim))
    else:
        last_start, last_end = block_bounds[-1]
        if z_dim - z_start > 512:
            # Add one more block
            block_bounds.append((z_start, z_dim))
        else:
            # Merge final slice range into the previous block
            block_bounds[-1] = (last_start, z_dim)

    return block_bounds

def get_tile_bounds(shape, block_size=1024, overlap=192):
    """
    Returns 2D tiles as
        (y0, y1, x0, x1)
    """
    ny, nx = shape

    y_blocks = get_block_bounds(ny, block_size, overlap)
    x_blocks = get_block_bounds(nx, block_size, overlap)

    tiles = []
    for y0, y1 in y_blocks:
        for x0, x1 in x_blocks:
            tiles.append((y0, y1, x0, x1))

    return tiles

def sam_boundaries(volume=None, volume_path=None, sam_checkpoint=None, boundaries_path=None, mask_path=None):
    TIC = time.time()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    # load mask data
    if mask_path:
        mask = load_data(mask_path)[0]
        argmin_z, argmax_z, argmin_y, argmax_y, argmin_x, argmax_x = reduce_blocksize(mask)
        del mask

    # load data
    if volume is None:
        volume = load_data(volume_path)[0]

    # crop volume to mask
    if mask_path:
        z_shape, y_shape, x_shape = volume.shape
        volume = volume[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x].copy()

    # data size
    zsh, ysh, xsh = volume.shape
    print('Data shape:', volume.shape)
    print('Data type:', volume.dtype)
    print('Blocks:', get_block_bounds(zsh))

    # initialize boundaries
    if rank==0:
        boundaries = np.zeros(volume.shape, np.uint8)

    # Load SAM model
    for m_type in ["vit_h", "vit_l", "vit_b"]:
        if m_type in sam_checkpoint:
            model_type = m_type
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    print(f"Rank {rank} using GPU {torch.cuda.current_device()}")

    # Create a mask generator
    mask_generator = SamAutomaticMaskGenerator(sam)

    # iterate over views
    for axis, sh in enumerate([zsh,ysh,xsh]):
        # iterate over slices
        print(" " * 40, end="\r")
        for slc in range(sh):
            if slc % size == rank:

                if rank==0:
                    print(f'Axis: {axis}, Slice: {slc+1}/{sh}', end='\r')

                # extract slice
                if axis==0:
                    slc_data = volume[slc].copy()
                elif axis==1:
                    slc_data = volume[:,slc].copy()
                elif axis==2:
                    slc_data = volume[:,:,slc].copy()

                # determine tiles
                tiles = get_tile_bounds(slc_data.shape)

                # initialize 2d boundaries
                slc_boundaries = np.zeros(slc_data.shape, np.uint32)

                # iterate over tiles
                for y0, y1, x0, x1 in tiles:
                    data = slc_data[y0:y1, x0:x1].copy()

                    # scale data
                    data = data.astype(np.float32)
                    data -= np.amin(data)
                    if np.amax(data)>0:
                        data /= np.amax(data)
                    data *= 255.0
                    data = data.astype(np.uint8)

                    # convert data to pseudo RGB
                    image = np.empty(data.shape + (3,), np.uint8)
                    for k in range(3):
                        image[:,:,k] = data

                    # run segmentation
                    masks = mask_generator.generate(image)

                    # determine buffer size
                    buff = 64
                    top    = buff if y0 > 0 else 0
                    bottom = buff if y1 < slc_data.shape[0] else 0
                    left   = buff if x0 > 0 else 0
                    right  = buff if x1 < slc_data.shape[1] else 0

                    # iterate over masks
                    for l, mask in enumerate(masks):
                        tile_boundaries = get_boundaries(mask["segmentation"].astype(np.uint8))
                        crop = tile_boundaries[top:top+(y1-bottom)-(y0+top), left:left+(x1-right)-(x0+left)]
                        slc_boundaries[y0+top:y1-bottom, x0+left:x1-right] += crop

                # binarize boundary
                slc_boundaries[slc_boundaries>0]=1
                slc_boundaries = slc_boundaries.astype(np.uint8)

                # add boundaries to volume
                if rank==0:
                    if axis==0:
                        boundaries[slc] += slc_boundaries
                    elif axis==1:
                        boundaries[:,slc] += slc_boundaries
                    elif axis==2:
                        boundaries[:,:,slc] += slc_boundaries

                    # gather results
                    for src in range(1, size):
                        if slc+src < sh:
                            comm.Recv([slc_boundaries, MPI.BYTE], source=src)
                            if axis==0:
                                boundaries[slc+src] += slc_boundaries
                            elif axis==1:
                                boundaries[:,slc+src] += slc_boundaries
                            elif axis==2:
                                boundaries[:,:,slc+src] += slc_boundaries

                else:
                    comm.Send([slc_boundaries.copy(), MPI.BYTE], dest=0)

    if rank==0:
        # calculation time
        print(int(round(time.time() - TIC)), 'sec')

        # restore original size
        if mask_path:
            output = np.zeros((z_shape, y_shape, x_shape), dtype=np.uint8)
            output[argmin_z:argmax_z, argmin_y:argmax_y, argmin_x:argmax_x] = boundaries
        else:
            output = boundaries

        # save data
        if boundaries_path:
            save_data(boundaries_path, output)
        return output

