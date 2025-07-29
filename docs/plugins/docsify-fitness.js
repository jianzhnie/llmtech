// docsify-fitness.js
window.$docsify.plugins.push(hook => {
  // 1. 挂载点：<div id="fitness-cal"></div>
  hook.doneEach(() => {
    const el = document.getElementById('fitness-cal');
    if (!el || el.__done) return;     // 防止重复渲染
    el.__done = true;

    // 2. 引入 vue & vue-hash-calendar
    loadScript('https://unpkg.com/vue@2/dist/vue.js', () => {
      loadScript('https://unpkg.com/vue-hash-calendar/lib/vue-hash-calendar.umd.min.js', () => {
        loadCSS('https://unpkg.com/vue-hash-calendar/lib/vue-hash-calendar.css');
        init(el);
      });
    });
  });

  // 3. 初始化组件
  function init(el) {
    new Vue({
      el,
      template: `
        <vue-hash-calendar
          :mark-date="marks"
          @click="onClick"
          format="YYYY-MM-DD"
          lang="zh"
          :show-today-button="false"
        />
      `,
      data: {
        marks: []   // 已打卡日期
      },
      created() {
        this.fetchRecords();
      },
      methods: {
        // 3-1 从 GitHub Issues 拉数据
        async fetchRecords() {
          const api = 'https://api.github.com/repos/jianzhnie/llmtech/issues';
          const res = await fetch(api);
          const issues = await res.json();
          this.marks = issues
            .filter(i => /^2\d{3}-\d{2}-\d{2}$/.test(i.title))
            .map(i => i.title);
        },
        // 3-2 点击日期 -> 打开打卡 Issue
        onClick(date) {
          const d = date.format('YYYY-MM-DD');
          const url = `https://github.com/jianzhnie/llmtech/issues/new?title=${d}&body=训练内容...`;
          window.open(url, '_blank');
        }
      }
    });
  }

  // 工具函数
  function loadScript(src, cb) {
    const s = document.createElement('script');
    s.src = src; s.onload = cb;
    document.head.appendChild(s);
  }
  function loadCSS(href) {
    const l = document.createElement('link');
    l.rel = 'stylesheet'; l.href = href;
    document.head.appendChild(l);
  }
});
