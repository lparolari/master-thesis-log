def fix_column_name(line):
    line = line.replace('stop sign,stopsign', 'stop sign/stopsign')
    line = line.replace('microwave,microwave oven', 'microwave/microwave oven')
    line = line.replace('refrigerator,fridge', 'refrigerator/fridge')
    line = line.replace('television,tv', 'television/tv')
    line = line.replace('sailboat,sail boat', 'sailboat/sail boat')
    line = line.replace('racket,racquet', 'racket/racquet')
    line = line.replace('headboard,head board', 'headboard/head board')
    line = line.replace('tennis racket,tennis racquet',
                        'tennis racket/tennis racquet')
    line = line.replace('skateboard,skate board', 'skateboard/skate board')
    line = line.replace('hot dog,hotdog', 'hot dog/hotdog')
    line = line.replace('surfboard,surf board', 'surfboard/surf board')
    line = line.replace('fire hydrant,hydrant', 'fire hydrant/hydrant')
    line = line.replace('suitcase,suit case', 'suitcase/suit case')
    line = line.replace('donut,doughnut', 'donut/doughnut')
    line = line.replace('sidewalk,side walk', 'sidewalk/side walk')
    line = line.replace('stove top,stovetop', 'stove top/stovetop')
    line = line.replace('nightstand,night stand', 'nightstand/night stand')
    line = line.replace('donuts,doughnuts', 'donuts/doughnuts')
    line = line.replace('lamp post,lamppost', 'lamp post/lamppost')
    line = line.replace('fire truck,firetruck', 'fire truck/firetruck')
    line = line.replace('tail light,taillight', 'tail light/taillight')
    line = line.replace('hot dogs,hotdogs', 'hot dogs/hotdogs')
    line = line.replace('tshirt,t-shirt,t shirt', 'tshirt/t-shirt/t shirt')
    line = line.replace('streetlight,street light', 'streetlight/street light')
    return line


if __name__ == '__main__':

    with open('flickr_train_class_freq.csv', 'r') as f:
        lines = f.readlines()

        # fix col name
        print(fix_column_name(lines[0]))

        # check col size
        for i in range(1, len(lines)):
            line = lines[i]
            line_size = len(line.split(','))
            if line_size != 1602:
                print(f"line {i} has wrong size ({line_size})")
                print("enter any key to continue")
                input()
