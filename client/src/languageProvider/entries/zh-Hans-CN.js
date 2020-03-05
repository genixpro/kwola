import appLocaleData from '@formatjs/intl-pluralrules/dist/locale-data/zh';
import zhMessages from '../locales/zh-Hans.json';

const ZhLan = {
  messages: {
    ...zhMessages,
  },
  antd: null,
  locale: 'zh-Hans-CN',
  data: appLocaleData,
};
export default ZhLan;
