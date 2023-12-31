import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "MoJing's Docs",
  lang: "zh-CN",
  description: "Mojing's documentation site",
  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }]
  ],
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: '/favicon.ico',
    nav: [
      { text: '主页', link: '/' },
      { text: '项目介绍', link: '/projects/' },
      { text: '个人博客', link: '/blogs/' }
    ],

    sidebar: [
      {
        text: '项目介绍',
        collapsed: false,
        items: [
          { text: '项目主页', link: '/projects/' },
          { text: '个人简历', link: '/projects/home-page' },
          { text: 'Tiktok复刻', link: '/projects/tiktok-clone' },
          { text: 'Tiktok复刻（API）', link: '/projects/tiktok-clone-api' },


        ]
      },
      {
        text: '个人博客',
        collapsed: false,
        items: [
          { text: '博客主页', link: '/blogs/' },
          {
            text: '模式识别',
            collapsed: false,
            items: [
              { text: '模板匹配算法', link: '/blogs/PatternRecognition/template-matching' },
              { text: 'SVM算法', link: '/blogs/PatternRecognition/SVM' },
              { text: '分类算法比较', link: '/blogs/PatternRecognition/multiple-classification-algorithms' },
              { text: '特征降维算法', link: '/blogs/PatternRecognition/feature-dimensionality-reduction' },
            ]
          },
        ]
      },
      {
        text: 'Examples',
        collapsed: true,
        items: [
          { text: 'Markdown Examples', link: '/markdown-examples' },
          { text: 'Runtime API Examples', link: '/api-examples' }
        ]
      }
    ],
    //lastUpdated: false,
    lastUpdatedText: '上次更新',
    returnToTopLabel: '返回顶部',
    darkModeSwitchLabel: '切换主题',
    sidebarMenuLabel: '目录',
    outline: {
      level: "deep", // 右侧大纲标题层级
      label: "文章目录", // 右侧大纲标题文本配置
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/mojing122' }
    ],
    docFooter:
    {
      prev: '上一篇',
      next: '下一篇'
    },
    footer: {
      message: 'MoJing - Keep Coding, Keep Fighting!',
      copyright: 'Copyright © 2023-Sha Zhiqing'
    },


    search: {
      provider: 'algolia',
      options: {
        appId: 'NMXRWCYMIW',
        apiKey: 'b13388e225641157f9a08bf09e4501e5',
        indexName: 'mojing',
        placeholder: '搜索文档',
        translations: {
          button: {
            buttonText: '搜索',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            searchBox: {
              resetButtonTitle: '清除查询条件',
              resetButtonAriaLabel: '清除查询条件',
              cancelButtonText: '取消',
              cancelButtonAriaLabel: '取消'
            },
            startScreen: {
              recentSearchesTitle: '搜索历史',
              noRecentSearchesText: '没有搜索历史',
              saveRecentSearchButtonTitle: '保存至搜索历史',
              removeRecentSearchButtonTitle: '从搜索历史中移除',
              favoriteSearchesTitle: '收藏',
              removeFavoriteSearchButtonTitle: '从收藏中移除'
            },
            errorScreen: {
              titleText: '无法获取结果',
              helpText: '你可能需要检查你的网络连接'
            },
            footer: {
              selectText: '选择',
              navigateText: '切换',
              closeText: '关闭',
              searchByText: '搜索提供者'
            },
            noResultsScreen: {
              noResultsText: '无法找到相关结果',
              suggestedQueryText: '你可以尝试查询',
              reportMissingResultsText: '你认为该查询应该有结果？',
              reportMissingResultsLinkText: '点击反馈'
            }
          }
        }
      }
    }
  }
})

