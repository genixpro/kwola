import { getDefaultPath } from '../../helpers/urlSync';

const options = [
  // {
  //   label: 'sidebar.dashboard',
  //   key: 'dashboard',
  // },
  {
    label: 'sidebar.newApplication',
    key: 'new-application',
  },
  {
    label: 'sidebar.listApplications',
    key: 'applications',
  },
];
const getBreadcrumbOption = () => {
  const preKeys = getDefaultPath();
  let parent, activeChildren;
  options.forEach(option => {
    if (preKeys[option.key]) {
      parent = option;
      (option.children || []).forEach(child => {
        if (preKeys[child.key]) {
          activeChildren = child;
        }
      });
    }
  });
  return { parent, activeChildren };
};
export default options;
export { getBreadcrumbOption };
