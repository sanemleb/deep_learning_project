class BgManager():
    detectron_map = {}

    def add_image(self, image_name, nobg_image):
        self.detectron_map[image_name] = nobg_image

    def get_image(self, image_name):
        if image_name in self.detectron_map.keys():
            return self.detectron_map.get(image_name)
        else: 
            return 'empty' 

    def get_img_no_bg(self, predictor, image):
        pred = predictor(image)
        mask = pred["instances"].pred_masks
        if (len(mask) == 0):
            return 'empty'
        i = len(mask[0][0])
        j = len(mask[0])
        test = image
        for j1 in range(j):
            for i1 in range(i):
                if (pred["instances"].pred_masks[0][j1][i1] == False):
                    test[j1,i1] = 2
        return test

