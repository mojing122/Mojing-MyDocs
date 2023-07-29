import{_ as a,a as s,b as n,c as e,d as t,e as l,f as o,g as p,h as i,i as r}from"./chunks/image-20230722165126280.bc678617.js";import{_ as c,o as h,c as m,V as d}from"./chunks/framework.ecf4210c.js";const E=JSON.parse('{"title":"基于Nuxt.js+Laravel复刻实现tiktok网页版","description":"","frontmatter":{},"headers":[],"relativePath":"projects/tiktok-clone.md","filePath":"projects/tiktok-clone.md"}'),u={name:"projects/tiktok-clone.md"},g=d('<h1 id="基于nuxt-js-laravel复刻实现tiktok网页版" tabindex="-1">基于Nuxt.js+Laravel复刻实现tiktok网页版 <a class="header-anchor" href="#基于nuxt-js-laravel复刻实现tiktok网页版" aria-label="Permalink to &quot;基于Nuxt.js+Laravel复刻实现tiktok网页版&quot;">​</a></h1><div style="margin-top:8px;"><img style="float:left;padding-right:5px;padding-bottom:8px;" src="https://img.shields.io/github/last-commit/mojing122/mojing-tiktok-clone-nuxt/master?logo=GitHub"><img style="float:left;padding-right:5px;padding-bottom:8px;" src="https://img.shields.io/github/repo-size/mojing122/mojing-tiktok-clone-nuxt"><img style="padding-bottom:8px;" src="https://img.shields.io/github/license/mojing122/mojing-tiktok-clone-nuxt"></div><p>本项目为前端部分，采用Nuxt.js框架实现，样式部分采用tailwindcss，后端api实现见项目<a href="https://github.com/mojing122/mojing-tiktok-clone-api-laravel" target="_blank" rel="noreferrer">mojing122/mojing-tiktok-clone-api-laravel (github.com)</a></p><p>项目参考John-Weeks-Dev的教程<a href="https://www.youtube.com/watch?v=CHSL0Btbj_o" target="_blank" rel="noreferrer">youtube.com/watch?v=CHSL0Btbj_o</a></p><p>预览：</p><p><img src="'+a+`" alt="ss-2023722"></p><p>网页预览：<a href="https://tiktok.mojing.live" target="_blank" rel="noreferrer">https://tiktok.mojing.live</a></p><p>Github：<a href="https://github.com/mojing122/mojing-tiktok-clone-nuxt/" target="_blank" rel="noreferrer">mojing122/mojing-tiktok-clone-nuxt (github.com)</a></p><p>Gitee：<a href="https://gitee.com/sha-zhiqing/mojing-tiktok-clone-nuxt/" target="_blank" rel="noreferrer">mojing-tiktok-clone-nuxt (gitee.com)</a></p><h2 id="实现功能" tabindex="-1">实现功能 <a class="header-anchor" href="#实现功能" aria-label="Permalink to &quot;实现功能&quot;">​</a></h2><ul><li>☑ 登录、注册</li><li>☑ 登录拦截器</li><li>☑ 首页 <ul><li>☑ 滚动页面自动播放/暂停</li><li>☑ 点赞</li><li>☑ 随机展示其他用户</li></ul></li><li>☑ 个人资料页 <ul><li>☑ 上传头像（需权限）</li><li>☑ 修改个人资料（需权限）</li><li>☑ 展示个人上传视频</li></ul></li><li>☑ 上传视频</li><li>☑ 视频详情页 <ul><li>☑ 上下切换视频</li><li>☑ 点赞、评论</li><li>☑ 删除视频（需权限）</li></ul></li><li>☐ ……</li></ul><h2 id="项目运行" tabindex="-1">项目运行 <a class="header-anchor" href="#项目运行" aria-label="Permalink to &quot;项目运行&quot;">​</a></h2><p>Look at the <a href="https://nuxt.com/docs/getting-started/introduction" target="_blank" rel="noreferrer">Nuxt 3 documentation</a> to learn more.</p><h3 id="setup" tabindex="-1">Setup <a class="header-anchor" href="#setup" aria-label="Permalink to &quot;Setup&quot;">​</a></h3><p>Make sure to install the dependencies:</p><div class="language-bash"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki material-theme-palenight"><code><span class="line"><span style="color:#676E95;font-style:italic;"># npm</span></span>
<span class="line"><span style="color:#FFCB6B;">npm</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">install</span></span>
<span class="line"></span>
<span class="line"><span style="color:#676E95;font-style:italic;"># pnpm</span></span>
<span class="line"><span style="color:#FFCB6B;">pnpm</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">install</span></span>
<span class="line"></span>
<span class="line"><span style="color:#676E95;font-style:italic;"># yarn</span></span>
<span class="line"><span style="color:#FFCB6B;">yarn</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">install</span></span></code></pre></div><h3 id="development-server" tabindex="-1">Development Server <a class="header-anchor" href="#development-server" aria-label="Permalink to &quot;Development Server&quot;">​</a></h3><p>Start the development server on <code>http://localhost:xxxx</code>:</p><div class="language-bash"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki material-theme-palenight"><code><span class="line"><span style="color:#676E95;font-style:italic;"># npm</span></span>
<span class="line"><span style="color:#FFCB6B;">npm</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">run</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">dev</span></span>
<span class="line"></span>
<span class="line"><span style="color:#676E95;font-style:italic;"># pnpm</span></span>
<span class="line"><span style="color:#FFCB6B;">pnpm</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">run</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">dev</span></span>
<span class="line"></span>
<span class="line"><span style="color:#676E95;font-style:italic;"># yarn</span></span>
<span class="line"><span style="color:#FFCB6B;">yarn</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">dev</span></span></code></pre></div><h3 id="production" tabindex="-1">Production <a class="header-anchor" href="#production" aria-label="Permalink to &quot;Production&quot;">​</a></h3><p>Build the application for production:</p><div class="language-bash"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki material-theme-palenight"><code><span class="line"><span style="color:#676E95;font-style:italic;"># npm</span></span>
<span class="line"><span style="color:#FFCB6B;">npm</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">run</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">build</span></span>
<span class="line"></span>
<span class="line"><span style="color:#676E95;font-style:italic;"># pnpm</span></span>
<span class="line"><span style="color:#FFCB6B;">pnpm</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">run</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">build</span></span>
<span class="line"></span>
<span class="line"><span style="color:#676E95;font-style:italic;"># yarn</span></span>
<span class="line"><span style="color:#FFCB6B;">yarn</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">build</span></span></code></pre></div><p>Locally preview production build:</p><div class="language-bash"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki material-theme-palenight"><code><span class="line"><span style="color:#676E95;font-style:italic;"># npm</span></span>
<span class="line"><span style="color:#FFCB6B;">npm</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">run</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">preview</span></span>
<span class="line"></span>
<span class="line"><span style="color:#676E95;font-style:italic;"># pnpm</span></span>
<span class="line"><span style="color:#FFCB6B;">pnpm</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">run</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">preview</span></span>
<span class="line"></span>
<span class="line"><span style="color:#676E95;font-style:italic;"># yarn</span></span>
<span class="line"><span style="color:#FFCB6B;">yarn</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">preview</span></span></code></pre></div><p>Check out the <a href="https://nuxt.com/docs/getting-started/deployment" target="_blank" rel="noreferrer">deployment documentation</a> for more information.</p><h2 id="项目截图" tabindex="-1">项目截图 <a class="header-anchor" href="#项目截图" aria-label="Permalink to &quot;项目截图&quot;">​</a></h2><h3 id="大屏效果" tabindex="-1">大屏效果： <a class="header-anchor" href="#大屏效果" aria-label="Permalink to &quot;大屏效果：&quot;">​</a></h3><p><img src="`+s+'" alt="image-20230722163609412"></p><p><img src="'+n+'" alt="image-20230722164303018"></p><p><img src="'+e+'" alt="image-20230722164432839"></p><p><img src="'+t+'" alt="image-20230722164503390"></p><p><img src="'+l+'" alt="image-20230722164551539"></p><h3 id="小屏效果" tabindex="-1">小屏效果： <a class="header-anchor" href="#小屏效果" aria-label="Permalink to &quot;小屏效果：&quot;">​</a></h3><table><thead><tr><th><img src="'+o+'" alt=""></th><th><img src="'+p+'" alt="image-20230722164938168"></th></tr></thead><tbody><tr><td><img src="'+i+'" alt="image-20230722165037412"></td><td><img src="'+r+'" alt="image-20230722165126280"></td></tr></tbody></table><h2 id="更多信息" tabindex="-1">更多信息 <a class="header-anchor" href="#更多信息" aria-label="Permalink to &quot;更多信息&quot;">​</a></h2><p>文档站：<a href="https://docs.mojing.live" target="_blank" rel="noreferrer">https://docs.mojing.live</a></p><h2 id="license" tabindex="-1">License <a class="header-anchor" href="#license" aria-label="Permalink to &quot;License&quot;">​</a></h2><p>Copyright [2023] [Sha Zhiqing]</p><p>Licensed under the Apache License, Version 2.0 (the &quot;License&quot;); you may not use this file except in compliance with the License.You may obtain a copy of the License at <a href="http://www.apache.org/licenses/LICENSE-2.0" target="_blank" rel="noreferrer">http://www.apache.org/licenses/LICENSE-2.0</a></p><p>Unless required by applicable law or agreed to in writing, softwaredistributed under the License is distributed on an &quot;AS IS&quot; BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.</p><h2 id="about-me" tabindex="-1">About me <a class="header-anchor" href="#about-me" aria-label="Permalink to &quot;About me&quot;">​</a></h2><p><img src="https://picgo-1304966930.cos.ap-nanjing.myqcloud.com/logo.png" alt="logo" style="zoom:15%;">MoJing 人工智能专业大三本科生</p>',42),y=[g];function C(b,k,f,_,A,v){return h(),m("div",null,y)}const j=c(u,[["render",C]]);export{E as __pageData,j as default};