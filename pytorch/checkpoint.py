import os

import torch


def resume(session, path=None):
    if path is None:
        directory = os.path.join(session.cfg['training']['checkpoint_directory'], session.cfg['name'])
        with open(os.path.join(directory, 'latest'), 'r') as fd:
            path = fd.read().strip()
    if not os.path.exists(path):
        raise OSError('"{}" does not exist.')
    print(f'Resuming from "{path}"...')
    checkpoint = torch.load(path)
    session.net.module.load_state_dict(checkpoint['net'])
    session.optimizer.load_state_dict(checkpoint['optimizer'])
    session.step = checkpoint['step']
    if session.swa.enabled and 'swa_net' in checkpoint:
        session.swa.net.module.load_state_dict(checkpoint['swa_net'])


def save(session):
    directory = os.path.join(session.cfg['training']['checkpoint_directory'], session.cfg['name'])
    if not os.path.exists(directory):
        os.makedirs(directory)
    # checkpoint
    checkpoint = {
        'net': session.net.module.state_dict(),
        'optimizer': session.optimizer.state_dict(),
        'step': session.step,
    }
    if session.swa.enabled:
        checkpoint['swa_net'] = session.swa.net.module.state_dict()
    filename = f'checkpoint-{session.step}.pth'
    path = os.path.join(directory, filename)
    torch.save(checkpoint, path)
    print(f'Checkpoint saved to "{path}"')
    # store path so that we know what checkpoint to resume from without specifying it
    with open(os.path.join(directory, 'latest'), 'w') as fd:
        fd.write(f'{path}\n')
    return
    # proto weights
    filename = f'net-{session.step}.pb.gz'
    path = os.path.join(directory, filename)
    session.net.module.export_proto(path)
    if session.swa.enabled:
        filename = f'net-swa-{session.step}.pb.gz'
        path = os.path.join(directory, filename)
        session.swa.net.module.export_proto(path)
