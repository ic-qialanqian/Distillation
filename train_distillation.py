import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import joint_transforms
from config import train_data
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from DANet import RGBD_sal
from torch.backends import cudnn
import torch.nn.functional as functional


cudnn.benchmark = True

torch.manual_seed(2021)
torch.cuda.set_device(0)

##########################hyperparameters###############################
ckpt_path = './model'
exp_name = ''
args = {
    'iter_num':20500,
    'train_batch_size': 4,
    'last_iter': 0,
    'lr': 1e-3,
    #'lr': 5e-5,
    'lr_decay': 0.9,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'snapshot': ''
}
##########################data augmentation###############################
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(256,256),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
target_transform = transforms.ToTensor()
##########################################################################
train_set = ImageFolder(train_data, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True)
criterion = nn.BCEWithLogitsLoss().cuda()
criterion_BCE = nn.BCELoss().cuda()
criterion_MAE = nn.L1Loss().cuda()
criterion_MSE = nn.MSELoss().cuda()
criterion_smoothL1 = nn.SmoothL1Loss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
KD_loss = nn.KLDivLoss().cuda()


def iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    #for i in range(0,b):
        #compute the IoU of the foreground
    Iand1 = torch.sum(target*pred)
    Ior1 = torch.sum(target) + torch.sum(pred)-Iand1
    IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        #IoU = IoU + (1-IoU1)

    return IoU1

   
        
def KD_KLDivLoss(Stu_output, Tea_output, temperature):
    T = temperature
    KD_loss = nn.KLDivLoss()(functional.log_softmax(Stu_output/T, dim=1), functional.softmax(Tea_output/T, dim=1)).cuda()
    KD_loss = KD_loss * T * T
    return KD_loss
    

        
    
def cross_entropy2d(input, target, temperature=1, weight=None, size_average=True):
    target = target.long()
    n, c, h, w = input.size()
    #print(input.size())
    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    input = input.view(-1, c)
    
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    
    T = temperature
    loss = functional.cross_entropy(input / T, target, weight=weight, size_average=size_average)
    # if size_average:
    #     loss /= mask.data.sum()
    return loss
    
    
    
def main():
    
    model_teacher = RGBD_sal().cuda()
    model = RGBD_sal1().cuda()
    
    
    model_teacher.load_state_dict(torch.load('./model/A2dele_depth_rfb/20500.pth'))
   
    net = model.train()
    
    
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])
    if len(args['snapshot']) > 0:
        print ('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    
    train(net, optimizer,model_teacher)
    

def train(net,optimizer,model_teacher):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, loss1_record, loss2_record,loss3_record,loss4_record,loss5_record,loss6_record,loss7_record,loss8_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter(),AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']
           
                                                            
            inputs, depth, labels= data
            labels[labels>0.5] = 1
            labels[labels!=1] = 0
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            depth = Variable(depth).cuda()
            labels = Variable(labels).cuda()
            
            n, c, h, w = inputs.size()
            
            teacher_out,d0,d1,d2,d3,d4 = model_teacher(inputs,depth)
            
            
            
            outputs=  net(inputs, depth)
            ##########loss#############
            optimizer.zero_grad()
            
            
            loss_teacher =cross_entropy2d(teacher_out.detach(), labels,temperature=20)
            
            
            
            
            LIPU_loss = KD_KLDivLoss(outputs, teacher_out.detach(),temperature=10)
            
            
            
            
            
           
            Dilation_teacher = functional.softmax(teacher_out, dim=1)
            Dilation_teacher = Dilation_teacher[:,1,:,:].view(n, 1, h, w)
            
            
            Dilation_output = functional.softmax(outputs, dim=1)
            Dilation_output = Dilation_output[:,1,:,:].view(n, 1, h, w)
            

            alpha = iou(Dilation_teacher.detach(),labels)
            
            beta = 1-iou(Dilation_output.detach(),labels)
            #p = 
            #g_mean = ((alpha)**p)* ((beta)**(1-p))
            #theta = torch.tanh(g_mean)
            theta=0.5
            
            
            labels0 = functional.interpolate(labels, size=16, mode='bilinear')
            labels1 = functional.interpolate(labels, size=17, mode='bilinear')
            labels2 = functional.interpolate(labels, size=32, mode='bilinear')
            labels3 = functional.interpolate(labels, size=64, mode='bilinear')
            labels4 = functional.interpolate(labels, size=128, mode='bilinear')
            

            
            
            loss = cross_entropy2d(outputs, labels,temperature=1)

            loss_adptative = theta*LIPU_loss + (1-theta)*(loss)
            
         
            
            total_loss =  loss_adptative
            
            total_loss.backward()
            optimizer.step()
            total_loss_record.update(total_loss.item(), batch_size)
            
            #loss1_record.update(loss_f0.item(), batch_size)
            #loss2_record.update(loss_f1.item(), batch_size)
            #loss3_record.update(loss_f2.item(), batch_size)
            #loss4_record.update(loss_f3.item(), batch_size)
            #loss5_record.update(loss_f4.item(), batch_size)
            loss6_record.update(loss.item(), batch_size)
            curr_iter += 1
            #############log###############
            if curr_iter %20500==0:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
           
            log = '[iter %d], [total loss %.5f], [LIPU_loss %.5f], [loss %.5f], [alpha %.5f], [beta %.5f], [theta %.5f]'  % \
                     (curr_iter, total_loss, LIPU_loss, loss,alpha,beta,theta)
            print(log)
            open(log_path, 'a').write(log + '\n')
            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
               
                return
            #############end###############

if __name__ == '__main__':
    main()
