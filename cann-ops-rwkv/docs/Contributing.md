## 了解行为准则
在参与贡献前，请了解[CANN开放项目行为准则](contributors/code-of-conduct.md)，后续您在CANN开放项目的活动（包括但不限于发表评论、提交Issue、发表wiki等）都请遵循此行为准则。

## 签署CLA

在参与项目贡献前，您需要签署CANN开放项目贡献者许可协议（CLA）。

请根据您的参与身份，选择签署个人CLA、公司CLA 或企业CLA，请点击[这里](https://clasign.osinfra.cn/sign/gitee_ascend-1720446461942705242)签署。

- 个人CLA：以个人身份参与贡献，请签署个人CLA。
- 企业管理员：以企业管理员的身份参与贡献，请签署企业管理员CLA。


## 参与贡献
在签署了CLA协议、找到了你想参与的开放项目后，就可以开始您的贡献之旅啦！贡献的方式有很多种，每一种贡献都将受到欢迎和重视。

### 提交Issue/处理Issue任务

- 找到Issue列表：
  
  在您感兴趣的CANN开放项目Gitee 主页内，点击“Issues”，即可找到 Issue 列表。

- 提交Issue
  
  如果您准备向社区上报Bug或者提交需求，或者为社区贡献自己的意见或建议，请在CANN开放项目对应的仓库上提交Issue。

  提交Issue请参考 [Issue 提交指南](contributors/issue-submit.md)。

- 参与Issue讨论

  每个Issue下面都支持开发者们交流讨论，如果您感兴趣，可以在评论区中发表自己的意见。

- 找到愿意处理的Issue

  如果您愿意处理其中的一个 issue，可以将它分配给自己。只需要在评论框内输入“/assign”或 “/assign @yourself”，机器人就会将问题分配给您，您的名字将显示在负责人列表里。

### 贡献编码

1. 准备CANN开发环境
  
   如果您想参与编码贡献，需要准备CANN开发环境，请参考每个开放项目的README.md，了解环境准备。

2. 了解CANN开放项目内的开发注意事项

   1）每个CANN开放项目使用的编码语言、开发编译环境等都可能存在差异，请参考每个开放项目中的README.md，了解编码贡献的一些要求。

   2）CANN开放项目软件编码遵循许可协议：CANN Open Software License Agreement Version 1.0，详细的协议说明请参见每个开放项目中的LICENSE文件，如果您贡献代码到CANN开放项目的源码仓，请遵循此协议。
   
     请在新建的源码文件（包括cpp、cc、h、py、sh等文件）头部增加如下声明：
   
     ```
     /**
      * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
      * This file is a part of the CANN Open Software.
      * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
      * Please refer to the License for details. You may not use this file except in compliance with the License.
      * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
      * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
      * See LICENSE in the root of the software repository for the full text of the License.
      */
     ```

3. 代码下载与贡献流程

   ![](https://obs-book.obs.cn-east-2.myhuaweicloud.com/cann-ops/images/contri-flow.png)

   1. 进行代码开发前，请先将需要参与开发的仓库fork到个人仓，然后将个人仓下载到本地。并在本地分支进行代码修改。
   2. 参考每个开放项目的说明文档，进行本地构建与验证。
   3. 代码验证满足贡献要求后，提交Pull-Request，将代码贡献到相应的开放项目。
   4. 请注意查看门禁测试结果，若未通过，请根据问题提示进行本地代码修改；若通过，此PR会被分配给commiter检视，请关注commiter的检视意见。
   5. 当您的PR检视通过后，代码会合入相应的开放项目。

   关于Gitee工作流的详细操作可参见[Gitee工作流说明](contributors/gitee-workflow.md)。
   
   当您在提交PR过程中遇到问题，常见问题的解决方法可参见[FAQs](contributors/infra-faqs.md)。