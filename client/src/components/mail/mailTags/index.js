import React from 'react';
import Icon from '../../uielements/icon';
import List, { ListItem, ListItemIcon, ListItemText } from './mailTags.style';

const tags = [
  'Friend',
  'Family',
  'Colleague',
  'Teachers',
  'Students',
  'ClassMates',
];
const tagsIcon = {
  Friend: 'people',
  Family: 'send',
  Colleague: 'drafts',
  Teachers: 'delete',
  Students: 'done',
  ClassMates: 'report',
};
const tagColor = [
  '#405dfe',
  '#90a0f0',
  '#03d16d',
  '#eb7345',
  '#2bc6da',
  '#776f6c',
];

function gettags(mails, filterAttr) {
  const tags = {};
  mails.forEach(mail => {
    if (mail.tags && mail.bucket === filterAttr.bucket) {
      mail.tags.split(' ').forEach(tag => (tags[tag] = 1));
    }
  });
  return tags;
}

export default function mailtags(
  mails,
  filterAction,
  filterAttr,
  onDrawerClose
) {
  const Tags = gettags(mails, filterAttr);
  const renderSingleTag = (tag, key) => {
    const onClick = () => {
      filterAction({ tag });
      if (onDrawerClose) {
        onDrawerClose();
      }
    };
    const selectedTag = tag === filterAttr.tag;
    const activeClass = selectedTag ? 'active' : '';
    const iconColor = tagColor[tags.findIndex(tags => tags === tag)];
    return (
      <ListItem
        key={`tag${key}`}
        onClick={onClick}
        className={`mailTag ${activeClass}`}
      >
        <ListItemIcon>
          <Icon style={{ color: iconColor }}>{tagsIcon[tag]}</Icon>
        </ListItemIcon>
        <ListItemText primary={tag} />
      </ListItem>
    );
  };
  return (
    <List>
      {Object.keys(Tags).map((tag, index) => renderSingleTag(tag, index))}
    </List>
  );
}
export { tags, tagColor, tagsIcon };
