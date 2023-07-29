---

title: 基于Nuxt.js+Laravel复刻实现tiktok网页版(API)

lang: zh-CN

---

# 基于Nuxt.js+Laravel复刻实现tiktok网页版(API)

<div style="margin-top:8px">
<img style="float: left;padding-right:5px;padding-bottom:8px" src="https://img.shields.io/github/last-commit/mojing122/mojing-tiktok-clone-api-laravel/master?logo=GitHub">
<img style="float: left;padding-right:5px;padding-bottom:8px" src="https://img.shields.io/github/repo-size/mojing122/mojing-tiktok-clone-api-laravel">
<img  style=";padding-bottom:8px" src="https://img.shields.io/github/license/mojing122/mojing-tiktok-clone-api-laravel">
</div>

本项目为mojing-tiktok-clone-nuxt的后端部分，采用Laravel框架+MySQL实现，前端见项目[mojing122/mojing-tiktok-clone-nuxt (github.com)](https://github.com/mojing122/mojing-tiktok-clone-nuxt/)

项目参考John-Weeks-Dev的教程[youtube.com/watch?v=CHSL0Btbj_o](https://www.youtube.com/watch?v=CHSL0Btbj_o)

预览：

![ss-2023722](/tiktok-clone/ss-2023722.gif)

网页预览：https://tiktok.mojing.live

Github：[mojing122/mojing-tiktok-clone-api-laravel (github.com)](https://github.com/mojing122/mojing-tiktok-clone-api-laravel)

Gitee：[mojing-tiktok-clone-api-laravel (gitee.com)](https://gitee.com/sha-zhiqing/mojing-tiktok-clone-api-laravel)

## 实现功能

- &#9745; 登录、注册
- &#9745; 登录拦截器
- &#9745; 首页
  - &#9745; 滚动页面自动播放/暂停
  - &#9745; 点赞
  - &#9745; 随机展示其他用户
- &#9745; 个人资料页
  - &#9745; 上传头像（需权限）
  - &#9745; 修改个人资料（需权限）
  - &#9745; 展示个人上传视频
- &#9745; 上传视频
- &#9745; 视频详情页
  - &#9745; 上下切换视频
  - &#9745; 点赞、评论
  - &#9745; 删除视频（需权限）
- &#9744; ……

## 项目运行

```bash
composer install 

cp .env.example .env 
```

修改.env中的DB_DATABASE、DB_USERNAME、DB_PASSWORD等配置

```
DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=[database name]
DB_USERNAME=[database username]
DB_PASSWORD=[database password]
```

连接数据库并运行服务

```bash
php artisan cache:clear 

composer dump-autoload 

php artisan key:generate

composer require laravel/breeze --dev

php artisan breeze:install 

php artisan serve
```

创建数据表

```bash
php artisan migrate
```

## 项目截图

### 大屏效果：

![image-20230722163609412](/tiktok-clone/image-20230722163609412.png)



![image-20230722164303018](/tiktok-clone/image-20230722164303018.png)

![image-20230722164432839](/tiktok-clone/image-20230722164432839.png)

![image-20230722164503390](/tiktok-clone/image-20230722164503390.png)

![image-20230722164551539](/tiktok-clone/image-20230722164551539.png)

### 小屏效果：

| ![](/tiktok-clone/image-20230722164757061.png)                | ![image-20230722164938168](/tiktok-clone/image-20230722164938168.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20230722165037412](/tiktok-clone/image-20230722165037412.png) | ![image-20230722165126280](/tiktok-clone/image-20230722165126280.png) |



## 更多信息

文档站：https://docs.mojing.live

## License

Copyright [2023] [Sha Zhiqing]

   Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, softwaredistributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## About me



<img src="https://picgo-1304966930.cos.ap-nanjing.myqcloud.com/logo.png" alt="logo" style="zoom:15%;" />MoJing 人工智能专业大三本科生