![输入图片说明](https://foruda.gitee.com/images/1732709982038009684/f1bee069_9519913.jpeg "首页banner.jpg")

## 🎯 项目介绍
cann-ops是昇腾与开发者共建的算子仓库，欢迎开发者学习、使用和贡献基于昇腾平台的系列算子代码。

## 🔍 仓库结构
cann-ops仓关键目录如下所示：
```
├── cmake
├── src // 算子源码目录
│ ├── common // 公共目录
│ ├── math // 数学库算子目录
│ │ └── add_custom // AddCustom算子目录
│ └── CMakeLists.txt
├── CMakeLists.txt
├── CMakePresets.json // 配置文件
├── LICENSE
├── README.md
└── build.sh // 算子编译脚本
```
## ⚡️ 快速上手
| 样例名称  | 样例介绍  | 开发语言  |
|---|---|---|
| [add_custom](./src/math/add_custom)  | 参考算子，基于Ascend C的vector加法算子  |  Ascend C |

## 📝 版本配套说明
请参考[CANN社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/softwareinst/instg/instg_0001.html)相关章节，对昇腾硬件、CANN软件及相应深度学习框架进行安装准备。

## 💬 社区交流
了解更多资源，欢迎访问[昇腾社区Ascend C](https://www.hiascend.com/ascend-c)

#### **📖 学习教程**
- **👉 [Ascend C算子开发（入门）](https://www.hiascend.com/developer/courses/detail/1691696509765107713)**
- **👉 [Ascend C算子开发（进阶）](https://www.hiascend.com/developer/courses/detail/1696414606799486977)**

#### **🔥系列直播 | 码力全开特辑**
- **直播平台**：【昇腾CANN】视频号、[B站【昇腾AI开发者】](https://space.bilibili.com/1190614918?spm_id_from=333.337.search-card.all.click)、[昇腾社区同步直播](https://www.hiascend.com/developer/activities/cann20251?tab=live)<br>
- **回放地址**：https://www.bilibili.com/video/BV1Xm421g7Ck?spm_id_from=333.788.videopod.sections&vd_source=c6dd04a5e994013eb635c132220694f7 <br>
- **直播预告**：<br>
![直播预告](https://obs-book.obs.cn-east-2.myhuaweicloud.com/cann-ops/resouce/CANNCAMP_Advance_03.png)

#### **🏅️ 项目发放**
- **众智任务**：[点击链接直达](https://www.hiascend.com/ecosystem/all-wisdom)
- **CANN训练营 | 实操挑战**：[点击链接直达](https://gitee.com/ascend/cann-ops/issues/IBOX5E?from=project-issue)
- **昇腾AI原生创新算子挑战赛**：[点击链接直达](https://www.hiascend.com/developer/ops)

#### **💌 联系我们**
若您对仓库的使用有任何建议和疑问，欢迎发送邮件到cann@huawei.com。<br>

 :globe_with_meridians: 网站：https://www.hiascend.com/software/cann <br>
 :mailbox_with_mail: 邮箱：cann@huawei.com <br>
 :speech_balloon: 论坛：https://www.hiascend.com/forum/forum-0106101385921175004-1.html <br>

## 🤝 共建伙伴
目前cann-ops开放仓库正在积极建设中，欢迎各社区伙伴积极参与社区共建，若您有意，请发送邮件至cann@huawei.com

## ⭐️ 贡献者
我们非常欢迎您为CANN贡献代码，也非常感谢您的反馈。<br>
![贡献者榜单-0415](https://obs-book.obs.cn-east-2.myhuaweicloud.com/cann-ops/resouce/contributor-0415.png)

## 📌 贡献指南
cann-ops仓欢迎广大开发者体验并参与贡献，在参与社区贡献之前。请参见[Contributing](docs/Contributing.md)了解行为准则，进行CLA协议签署，以及参与源码仓贡献的详细流程。<br>

**针对cann-ops仓，开发者准备本地代码与提交PR时需要重点关注如下几点**：<br>

1.提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。<br>
2.若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为“简单的bug修复”，亦可通过提交Issue进行方案讨论。

## 📄 许可证书
[CANN Open Software License Agreement Version 1.0](LICENSE)
