""" Hand in Depth
    https://github.com/xkunwu/depth-hand
"""
import os
import sys
from importlib import import_module
import argparse
import logging
from utils.coder import file_pack

model_ref = {
    'super_edt3': 'EDT3',
    'super_ov3edt2m': 'MV-CR w/ surface distance (weighted)',
    'super_ov3dist2': 'MV-CR w/ Euclidean distance',
    'super_ov3edt2': 'MV-CR w/ surface distance',
    'super_edt2m': '2D CR w/ surface distance (weighted)',
    'super_edt2': '2D CR w/ surface distance',
    'super_dist3': '3D CR w/ Euclidean distance',
    'voxel_regre': '3D CR w/ offset',
    'voxel_offset': '3D offset regression',
    'super_vxhit': '3D CR w/ detection',
    'voxel_detect': 'Moon et al. (CVPR\'18)',
    'super_dist2': '2D CR w/ Euclidean distance',
    'super_udir2': '2D CR w/ offset',
    'super_hmap2': '2D CR w/ heatmap',
    'dense_regre': '2D offset regression',
    'direc_tsdf': 'Ge et al. (CVPR\'17)',
    'trunc_dist': '3D truncated Euclidean distance',
    'base_conv3': '3D CR',
    'base_conv3_inres': '3D CR w/ inception-resnet',
    'ortho3view': 'Ge et al. (CVPR\'16)',
    'base_clean': '2D CR',
    'base_regre': '2D CR-background',
    'base_clean_inres': '2D CR w/ inception-resnet',
    'base_regre_inres': '2D CR-background w/ inception-resnet',
    'base_clean_hg': '2D CR w/ hourglass',
    'base_regre_hg': '2D CR-background w/ hourglass',
    'localizer3': '3D localizer',
    'localizer2': '2D localizer',
}

model_map = {
    'super_edt3': 'model.super_edt3',
    'super_ov3edt2m': 'model.super_ov3edt2m',
    'super_ov3dist2': 'model.super_ov3dist2',
    'super_ov3edt2': 'model.super_ov3edt2',
    'super_edt2m': 'model.super_edt2m',
    'super_edt2': 'model.super_edt2',
    'super_dist3': 'model.super_dist3',
    'voxel_regre': 'model.voxel_regre',
    'voxel_offset': 'model.voxel_offset',
    'super_vxhit': 'model.super_vxhit',
    'voxel_detect': 'model.voxel_detect',
    'super_dist2': 'model.super_dist2',
    'super_udir2': 'model.super_udir2',
    'super_hmap2': 'model.super_hmap2',
    'dense_regre': 'model.dense_regre',
    'direc_tsdf': 'model.direc_tsdf',
    'trunc_dist': 'model.trunc_dist',
    'base_conv3': 'model.base_conv3',
    'base_conv3_inres': 'model.base_inres',
    'ortho3view': 'model.ortho3view',
    'base_clean': 'model.base_clean',
    'base_regre': 'model.base_regre',
    'base_clean_inres': 'model.base_inres',
    'base_regre_inres': 'model.base_inres',
    'base_clean_hg': 'model.base_hourglass',
    'base_regre_hg': 'model.base_hourglass',
    'localizer3': 'model.localizer3',
    'localizer2': 'model.localizer2',
}


class args_holder:
    """ this class holds all arguments, and provides parsing functionality """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.logFormatter = logging.Formatter(
            # '%(asctime)s [%(levelname)-5.5s]  %(message)s (%(filename)s:%(lineno)s)',
            '%(asctime)s [%(levelname)-5.5s]  %(message)s',
            datefmt='%y-%m-%d %H:%M:%S')

        # directories
        home_dir = os.path.expanduser('~')
        this_dir = os.path.dirname(os.path.abspath(__file__))
        proj_root = os.path.abspath(os.path.join(this_dir, os.pardir))
        self.parser.add_argument(
            '--data_root', default=os.path.join(home_dir, 'data'),
            help='root dir of all data sets [default: data]')
        self.parser.add_argument(
            '--data_name', default='hands17',
            help='name of data set and its dir [default: hands17]')
        self.parser.add_argument(
            '--num_eval', type=int, default=None,
            help='number of evaluations [default: None]')
        # self.parser.add_argument(
        #     '--out_root', default=os.path.join(proj_root, 'output'),
        #     help='Output dir [default: output]')
        self.parser.add_argument(
            '--out_root', default=os.path.join(home_dir, 'data', 'univue', 'output'),
            help='Output dir [default: output]')
        self.parser.add_argument(
            '--retrain', dest='retrain', action='store_true',
            help='retrain the model')
        self.parser.set_defaults(retrain=False)
        self.parser.add_argument(
            '--mode', default='train',
            help='programm mode [default: train], from \
            [train, detect]')
        self.parser.add_argument(
            '--show_draw', default=False,
            help='allow popping out drawing window')
        self.parser.add_argument(
            '--print_models', dest='print_models', action='store_true',
            help='print the model maps and exit')
        self.parser.set_defaults(print_models=False)
        self.parser.add_argument(
            '--print_model_prereq', dest='print_model_prereq', action='store_true',
            help='print the prepared data dependency for the current model class and exit')
        self.parser.set_defaults(print_model_prereq=False)
        self.parser.add_argument(
            '--print_data_deps', dest='print_data_deps', action='store_true',
            help='print the prepared data dependency structure and exit')
        self.parser.set_defaults(print_data_deps=False)

        # system parameters
        self.parser.add_argument(
            '--gpu_id', type=int, default=0,
            help='GPU to use [default: GPU 0]')
        # [base_regre, base_clean, ortho3view, base_conv3, trunc_dist]
        self.parser.add_argument(
            # '--model_name', default='ortho3view',
            '--model_name', default='base_clean',
            help='Model name [default: base_clean], from \
            [base_regre, base_clean, ortho3view, base_conv3, trunc_dist]')
        self.parser.add_argument(
            '--model_desc', default='',
            help='variant version description [default: [empty]]'
        )
        self.parser.add_argument(
            '--localizer_name', default='localizer2',
            help='localize hand region [default: localizer2]'
        )

        # learning parameters
        self.parser.add_argument(
            '--max_epoch', type=int, default=10,
            help='Epoch to run [default: 10]')
        self.parser.add_argument(
            '--valid_stop', type=float, default=0.1,
            help='stop training early when validation increased [default: 0.1]')
        self.parser.add_argument(
            '--batch_size', type=int, default=50,
            help='Batch size during training [default: 50]')
        # self.parser.add_argument(
        #     '--optimizer', default='adam',
        #     help='Only using adam currently [default: adam]')
        self.parser.add_argument(
            '--bn_momentum', type=float, default=0.8,
            help='Initial batch normalization momentum [default: 0.8]')
        # exponential moving average is actually alpha filter in signal processing,
        # the time to converge is approximately 1/(1-decay) steps of train.
        # For decay=0.999, you need 1/0.001=1000 steps to converge.
        # Lower `decay` value (recommend trying `decay`=0.9) if model experiences
        # reasonably good training performance but poor validation and/or test performance.
        self.parser.add_argument(
            '--bn_decay', type=float, default=0.9997,
            help='decay rate during batch normalization [default: 0.9997]')
        self.parser.add_argument(
            '--regu_scale', type=float, default=0.01,
            help='regularization scale [default: 0.01]')
        self.parser.add_argument(
            '--learning_rate', type=float, default=0.001,
            help='Initial learning rate [default: 0.001]')
        self.parser.add_argument(
            '--decay_step', type=int, default=1000000,
            # twice of 1M (1e6) dataset, will be divided by batch size below
            help='Decay step for lr decay [default: 2e6]')
        self.parser.add_argument(
            '--decay_rate', type=float, default=0.94,
            # fast decay, as using adaptive optimizer
            help='Decay rate for lr decay [default: 0.94]')

        # detection & tracking
        self.parser.add_argument(
            '--show_debug', default=False,
            help='show debug figures and additional info')
        self.parser.add_argument(
            '--read_stream', default=False,
            help='read raw stream')
        self.parser.add_argument(
            '--save_stream', default=False,
            help='save raw stream')
        self.parser.add_argument(
            '--save_det', default=False,
            help='save detection results')
        self.parser.add_argument(
            '--use_zed', default=False,
            help='use Zed camera')


    def make_new_log(self):
        self.args.log_dir = os.path.join(self.args.out_dir, 'log')
        blinks = os.path.join(self.args.log_dir, 'blinks')
        if not os.path.exists(blinks):
            os.makedirs(blinks)
        log_dir_ln = os.path.join(
            blinks, self.args.model_name + self.args.model_desc)
        if (not os.path.exists(log_dir_ln)) or self.args.retrain:
            from datetime import datetime
            log_time = datetime.now().strftime('%y%m%d-%H%M%S')
            # git_hash = subprocess.check_output(
            #     ['git', 'rev-parse', '--short', 'HEAD'])
            self.args.log_dir_t = os.path.join(
                self.args.log_dir, 'log-{}-{}'.format(
                    self.args.model_name + self.args.model_desc,
                    log_time)
            )
            os.makedirs(self.args.log_dir_t)
            os.symlink(self.args.log_dir_t, log_dir_ln + '-tmp')
            os.rename(log_dir_ln + '-tmp', log_dir_ln)
        else:
            self.args.log_dir_t = os.readlink(log_dir_ln)

    def make_central_logging(self):
        logger = logging.getLogger('univue')
        logger.setLevel(logging.INFO)
        fileHandler = logging.FileHandler(
            os.path.join(
                self.args.log_dir_t, 'univue.log'),
            mode='a'  # write seperately for multi-processing
        )
        fileHandler.setFormatter(self.logFormatter)
        logger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler(stream=sys.stdout)
        consoleHandler.setFormatter(self.logFormatter)
        logger.addHandler(consoleHandler)
        self.args.logger = logger

    def make_logging(self):
        self.make_central_logging()
        logger = logging.getLogger('train')
        logger.setLevel(logging.INFO)
        if self.args.retrain:
            fileHandler = logging.FileHandler(
                os.path.join(self.args.log_dir_t, 'train.log'),
                mode='w'
            )
        else:
            fileHandler = logging.FileHandler(
                os.path.join(self.args.log_dir_t, 'train.log'),
                mode='a'
            )
        fileHandler.setFormatter(self.logFormatter)
        logger.addHandler(fileHandler)
        # if 0 < self.args.gpu_id:  # do not messy console
        #     return
        consoleHandler = logging.StreamHandler(stream=sys.stdout)
        consoleHandler.setFormatter(self.logFormatter)
        logger.addHandler(consoleHandler)

    def append_log(self):
        # logger = logging.getLogger('univue')
        # for handler in logger.handlers[:]:
        #     handler.close()
        #     logger.removeHandler(handler)
        with open(os.path.join(
                self.args.log_dir, 'univue.log'), mode='a') as outfile:
            with open(os.path.join(
                    self.args.log_dir_t, 'univue.log'), mode='r') as infile:
                outfile.write(infile.read())
        # self.make_central_logging()

    @staticmethod
    def write_args(args):
        import inspect
        with open(os.path.join(args.log_dir_t, 'args.txt'), 'a') as writer:
            writer.write('###################################\n')
            for arg in vars(args):
                att = getattr(args, arg)
                if inspect.ismodule(att) or inspect.isclass(att):
                    continue
                writer.write('--{}={}\n'.format(arg, att))
                # print(arg, getattr(args, arg))
            writer.write('###################################\n')
            args.model_inst.write_args(writer)

    # parse arguments, and only perform very basic managements
    def parse_args(self):
        try:
            self.args = self.parser.parse_args()
        except:
            self.args = None
            self.parser.print_help()
            return False
        if self.args.print_models:
            print('Currently implemented model list:')
            for k, v in model_map.items():
                print(k, " --> ", v, ": ", model_ref[k])
            return False
        self.args.filepack = file_pack()  # central file pack
        self.args.data_dir = os.path.join(
            self.args.data_root,
            self.args.data_name)
        self.args.out_dir = os.path.join(
            self.args.out_root,
            self.args.data_name
        )
        self.args.stream_dir = os.path.join(
            self.args.out_dir,
            'capture', 'stream'
        )
        if not os.path.exists(self.args.stream_dir):
            os.makedirs(self.args.stream_dir)
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.args:
            self.args.filepack = None
        logger = logging.getLogger('univue')
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logger = logging.getLogger('train')
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logging.shutdown()
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            os._exit(0)
        return self

    # this is called after hard coded parameter tweakings
    def create_instance(self):
        # logging system
        self.args.log_dir = os.path.join(self.args.out_dir, 'log')
        self.make_new_log()
        if not os.path.exists(os.path.join(
                self.args.log_dir_t, 'model.ckpt.meta')):
            self.args.retrain = True
        self.make_logging()
        self.args.logger.info('######## {} [{}] ########'.format(
            self.args.data_name,
            self.args.model_name + self.args.model_desc))

        # create model instance now
        if self.args.mode in ['train', 'detect']:
            self.args.model_class = getattr(
                import_module(model_map[self.args.model_name]),
                self.args.model_name
            )
            self.args.model_inst = self.args.model_class(self.args)
        else:
            raise ValueError('mode (%s) not recognized', self.args.mode)
        # if 'detect' == self.args.mode:
        #     self.args.localizer_class = getattr(
        #         import_module(model_map[self.args.localizer_name]),
        #         self.args.localizer_name
        #     )
        #     self.args.localizer_inst = self.args.localizer_class(self.args)
        # model instance has a chance to tweak parameters if necessary
        self.args.model_inst.tweak_arguments(self.args)
        self.args.decay_step //= self.args.batch_size

        # create data instance
        self.args.data_module = import_module(
            'data.' + self.args.data_name)
        self.args.data_provider = import_module(
            'data.' + self.args.data_name + '.provider')
        # self.args.data_draw = import_module(
        #     'data.' + self.args.data_name + '.draw')
        # self.args.data_eval = import_module(
        #     'data.' + self.args.data_name + '.eval')
        # self.args.data_ops = import_module(
        #     'data.' + self.args.data_name + '.ops')
        # self.args.data_io = import_module(
        #     'data.' + self.args.data_name + '.io')
        self.args.data_draw = getattr(
            import_module('data.' + self.args.data_name + '.draw'),
            'draw')
        self.args.data_eval = getattr(
            import_module('data.' + self.args.data_name + '.eval'),
            'eval')
        self.args.data_io = getattr(
            import_module('data.' + self.args.data_name + '.io'),
            'io')
        self.args.data_ops = getattr(
            import_module('data.' + self.args.data_name + '.ops'),
            'ops')
        # self.args.data_provider = getattr(
        #     import_module('data.' + self.args.data_name + '.provider'),
        #     'provider')
        self.args.data_class = getattr(
            import_module('data.' + self.args.data_name + '.holder'),
            self.args.data_name + 'holder'
        )
        self.args.data_inst = self.args.data_class(self.args)
        self.args.data_inst.init_data()

        # bind data instance to model instance
        self.args.model_inst.receive_data(self.args.data_inst, self.args)

        # asked for information?
        if self.args.print_model_prereq:
            print('Prepared dependency for {}: {}'.format(
                self.args.model_name,
                set(self.args.model_inst.store_name.values())))
            return False
        if self.args.print_data_deps:
            print('Data dependency hierarchy for {}:'.format(self.args.data_name))
            for k, v in self.args.data_inst.store_precon.items():
                print(k, " --> ", v)
            return False

        self.write_args(self.args)
        return True


if __name__ == "__main__":
    # python args_holder.py --batch_size=16
    with args_holder() as argsholder:
        if not argsholder.parse_args():
            os._exit(0)
        ARGS = argsholder.args
        if not argsholder.create_instance():
            os._exit(0)
