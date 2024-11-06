import os

from dataload import dataset_load

def get_path(dataset,basepath,source=None,target=None):
    s_path = None
    t_path = None
    if (dataset == 'visda'):
            s_path = os.path.join(basepath, "VisDA/train")
            t_path = os.path.join(basepath, "VisDA/validation")
    elif (dataset == 'office31'):
        if source is not None:
            s_path = os.path.join(basepath, "office31")
            if (source == 'A'):
                s_path = os.path.join(s_path, "amazon")
            elif (source == 'D'):
                s_path = os.path.join(s_path, "dslr")
            elif (source == 'W'):
                s_path = os.path.join(s_path, "webcam")

        if target is not None:
            t_path = os.path.join(basepath, "office31")
            if (target == 'A'):
                t_path = os.path.join(t_path, "amazon")
            elif (target == 'D'):
                t_path = os.path.join(t_path, "dslr")
            elif (target == 'W'):
                t_path = os.path.join(t_path, "webcam")

    elif (dataset == "officehome"):
        if source is not None:
            s_path = os.path.join(basepath, "office_home")
            if (source == 'A'):
                s_path = os.path.join(s_path, "Art")
            elif (source == 'C'):
                s_path = os.path.join(s_path, "Clipart")
            elif (source == 'P'):
                s_path = os.path.join(s_path, "Product")
            elif(source == "R"):
                s_path = os.path.join(s_path,"Real_World")

        if target is not None:
            t_path = os.path.join(basepath, "office_home")
            if (target == 'A'):
                t_path = os.path.join(t_path, "Art")
            elif (target == 'C'):
                t_path = os.path.join(t_path, "Clipart")
            elif (target == 'P'):
                t_path = os.path.join(t_path, "Product")
            elif(target == "R"):
                t_path = os.path.join(t_path,"Real_World")
    elif (dataset == "pacs"):
        if source is not None:
            s_path = os.path.join(basepath, "PACS")
            if (source == 'A'):
                s_path = os.path.join(s_path, "art_painting")
            elif (source == 'C'):
                s_path = os.path.join(s_path, "cartoon")
            elif (source == 'P'):
                s_path = os.path.join(s_path, "photo")
            elif(source == "S"):
                s_path = os.path.join(s_path,"sketch")

        if target is not None:
            t_path = os.path.join(basepath, "PACS")
            if (target == 'A'):
                t_path = os.path.join(t_path, "art_painting")
            elif (target == 'C'):
                t_path = os.path.join(t_path, "cartoon")
            elif (target == 'P'):
                t_path = os.path.join(t_path, "photo")
            elif(target == "S"):
                t_path = os.path.join(t_path,"sketch")
    elif (dataset == "image_clef"):
        if source is not None:
            s_path = os.path.join(basepath, "image_CLEF")
            if (source == 'B'):
                s_path = os.path.join(s_path, "b")
            elif (source == 'C'):
                s_path = os.path.join(s_path, "c")
            elif (source == 'I'):
                s_path = os.path.join(s_path, "i")
            elif(source == "P"):
                s_path = os.path.join(s_path,"p")

        if target is not None:
            t_path = os.path.join(basepath, "image_CLEF")
            if (target == 'B'):
                t_path = os.path.join(t_path, "b")
            elif (target == 'C'):
                t_path = os.path.join(t_path, "c")
            elif (target == 'I'):
                t_path = os.path.join(t_path, "i")
            elif(target == "P"):
                t_path = os.path.join(t_path,"p")
    elif (dataset == "domain_net"):
        if source is not None:
            s_path = os.path.join(basepath,"domain_net")
            print(source)
            if (source == 'sketch'):
                s_path = os.path.join(s_path,"sketch")
            elif (source == "real"):
                s_path = os.path.join(s_path,"real")
            elif (source == "quickdraw"):
                s_path = os.path.join(s_path,"quickdraw")
            elif (source == "painting"):
                s_path = os.path.join(s_path,"painting")
            elif (source == "infograph"):
                s_path = os.path.join(s_path,"infograph")
            elif (source == "clipart"):
                s_path = os.path.join(s_path,"clipart")
        if target is not None:
            t_path = os.path.join(basepath,"domain_net")
            print(target)
            if (target == 'sketch'):
                t_path = os.path.join(t_path,"sketch")
            elif (target == "real"):
                t_path = os.path.join(t_path,"real")
            elif (target == "quickdraw"):
                t_path = os.path.join(t_path,"quickdraw")
            elif (target == "painting"):
                t_path = os.path.join(t_path,"painting")
            elif (target == "infograph"):
                t_path = os.path.join(t_path,"infograph")
            elif (target == "clipart"):
                t_path = os.path.join(t_path,"clipart")
    return s_path, t_path

def get_dataloader(batch_size, dataset, train, source=None, target=None, basepath="../dataset", drop_last=False,
                   transform=None,shuffle=True,num_workers=1):
    source_loader,target_loader = None,None
    if (dataset == 'visda'):
        if source is not None:
            s_path = os.path.join(basepath,"VisDA/train")
            source_loader = dataset_load.load_visda_dataset(s_path, train=train, batch_size=batch_size,
                                                             drop_last=drop_last, transform=transform, shuffle=shuffle,
                                                             num_workers=num_workers)
        if target is not None:
            t_path = os.path.join(basepath, "VisDA/validation")
            target_loader = dataset_load.load_visda_dataset(t_path, train=train, batch_size=batch_size,
                                                             drop_last=drop_last, transform=transform, shuffle=shuffle,
                                                             num_workers=num_workers)
    elif (dataset == 'office31'):
        if source is not None:
            s_path = os.path.join(basepath, "office31")
            if (source == 'A'):
                s_path = os.path.join(s_path, "amazon")
            elif (source == 'D'):
                s_path = os.path.join(s_path, "dslr")
            elif (source == 'W'):
                s_path = os.path.join(s_path, "webcam")
            source_loader = dataset_load.load_office_dataset(s_path, train=train, batch_size=batch_size,
                                                             drop_last=drop_last, transform=transform, shuffle=shuffle,
                                                             num_workers=num_workers)
        if target is not None:
            t_path = os.path.join(basepath, "office31")
            if (target == 'A'):
                t_path = os.path.join(t_path, "amazon")
            elif (target == 'D'):
                t_path = os.path.join(t_path, "dslr")
            elif (target == 'W'):
                t_path = os.path.join(t_path, "webcam")
            target_loader = dataset_load.load_office_dataset(t_path, train=train, batch_size=batch_size,
                                                             drop_last=drop_last, transform=transform, shuffle=shuffle,
                                                             num_workers=num_workers)
    elif (dataset == "officehome"):
        if source is not None:
            s_path = os.path.join(basepath, "office_home")
            if (source == 'A'):
                s_path = os.path.join(s_path, "Art")
            elif (source == 'C'):
                s_path = os.path.join(s_path, "Clipart")
            elif (source == 'P'):
                s_path = os.path.join(s_path, "Product")
            elif(source == "R"):
                s_path = os.path.join(s_path,"Real_World")
            source_loader = dataset_load.load_office_dataset(s_path, train=train, batch_size=batch_size,
                                                             drop_last=drop_last, transform=transform, shuffle=shuffle,
                                                             num_workers=num_workers)
        if target is not None:
            t_path = os.path.join(basepath, "office_home")
            if (target == 'A'):
                t_path = os.path.join(t_path, "Art")
            elif (target == 'C'):
                t_path = os.path.join(t_path, "Clipart")
            elif (target == 'P'):
                t_path = os.path.join(t_path, "Product")
            elif(target == "R"):
                t_path = os.path.join(t_path,"Real_World")
            target_loader = dataset_load.load_office_dataset(t_path, train=train, batch_size=batch_size,
                                                             drop_last=drop_last, transform=transform, shuffle=shuffle,
                                                             num_workers=num_workers)

    elif (dataset == "pacs"):
        if source is not None:
            s_path = os.path.join(basepath, "PACS")
            if (source == 'A'):
                s_path = os.path.join(s_path, "art_painting")
            elif (source == 'C'):
                s_path = os.path.join(s_path, "cartoon")
            elif (source == 'P'):
                s_path = os.path.join(s_path, "photo")
            elif (source == "S"):
                s_path = os.path.join(s_path, "sketch")
            source_loader = dataset_load.load_all_can_use(s_path, train=train, batch_size=batch_size,
                                                             drop_last=drop_last, transform=transform, shuffle=shuffle,
                                                             num_workers=num_workers)
        if target is not None:
            t_path = os.path.join(basepath, "PACS")
            if (target == 'A'):
                t_path = os.path.join(t_path, "art_painting")
            elif (target == 'C'):
                t_path = os.path.join(t_path, "cartoon")
            elif (target == 'P'):
                t_path = os.path.join(t_path, "photo")
            elif (target == "S"):
                t_path = os.path.join(t_path, "sketch")
            target_loader = dataset_load.load_all_can_use(t_path, train=train, batch_size=batch_size,
                                                             drop_last=drop_last, transform=transform, shuffle=shuffle,
                                                             num_workers=num_workers)

    elif (dataset == "image_clef"):
        if source is not None:
            s_path = os.path.join(basepath, "image_CLEF")
            if (source == 'B'):
                s_path = os.path.join(s_path, "b")
            elif (source == 'C'):
                s_path = os.path.join(s_path, "c")
            elif (source == 'I'):
                s_path = os.path.join(s_path, "i")
            elif (source == "P"):
                s_path = os.path.join(s_path, "p")
            source_loader = dataset_load.load_all_can_use(s_path, train=train, batch_size=batch_size,
                                                             drop_last=drop_last, transform=transform, shuffle=shuffle,
                                                             num_workers=num_workers)
        if target is not None:
            t_path = os.path.join(basepath, "image_CLEF")
            if (target == 'B'):
                t_path = os.path.join(t_path, "b")
            elif (target == 'C'):
                t_path = os.path.join(t_path, "c")
            elif (target == 'I'):
                t_path = os.path.join(t_path, "i")
            elif (target == "P"):
                t_path = os.path.join(t_path, "p")
            target_loader = dataset_load.load_all_can_use(t_path, train=train, batch_size=batch_size,
                                                          drop_last=drop_last, transform=transform, shuffle=shuffle,
                                                          num_workers=num_workers)
    elif (dataset == "domain_net"):
        if source is not None:
            s_path = os.path.join(basepath,"domain_net")
            if (source == 'sketch'):
                s_path = os.path.join(s_path,"sketch")
            elif (source == "real"):
                s_path = os.path.join(s_path,"real")
            elif (source == "quickdraw"):
                s_path = os.path.join(s_path,"quickdraw")
            elif (source == "painting"):
                s_path == os.path.join(s_path,"painting")
            elif (source == "infograph"):
                s_path == os.path.join(s_path,"infograph")
            elif (source == "clipart"):
                s_path == os.path.join(s_path,"clipart")
            source_loader = dataset_load.load_all_can_use(s_path, train=train, batch_size=batch_size,
                                                 drop_last=drop_last, transform=transform, shuffle=shuffle,
                                                 num_workers=num_workers)  
            
        if target is not None:
            t_path = os.path.join(basepath,"domain_net")
            if (target == 'sketch'):
                t_path = os.path.join(t_path,"sketch")
            elif (target == "real"):
                t_path = os.path.join(t_path,"real")
            elif (target == "quickdraw"):
                t_path = os.path.join(t_path,"quickdraw")
            elif (target == "painting"):
                t_path == os.path.join(t_path,"painting")
            elif (target == "infograph"):
                t_path == os.path.join(t_path,"infograph")
            elif (target == "clipart"):
                t_path == os.path.join(t_path,"clipart")
            target_loader = dataset_load.load_all_can_use(t_path, train=train, batch_size=batch_size,
                                              drop_last=drop_last, transform=transform, shuffle=shuffle,
                                              num_workers=num_workers)

    return source_loader, target_loader