import React from 'react';
import Icon from '../../uielements/icon';
import List, { ListItem, ListItemIcon, ListItemText } from './mailBucket.style';

const buckets = ['Inbox', 'Sent', 'Done', 'Drafts', 'Bin', 'Spam'];

const bucketIcon = {
  Inbox: 'inbox',
  Sent: 'send',
  Done: 'done',
  Drafts: 'drafts',
  Bin: 'delete',
  Spam: 'report',
};

const bucketColor = [
  '#3949AB',
  '#2296f3',
  '#1eaa70',
  '#757575',
  '#757575',
  '#757575',
];

function getUnread(mails) {
  const unread = {};
  mails.forEach(mail => {
    if (!unread[mail.bucket]) {
      unread[mail.bucket] = 0;
    }
    if (!mail.read) {
      unread[mail.bucket] += 1;
    }
  });
  return unread;
}

export default function mailbuckets(
  mails,
  filterAction,
  filterAttr,
  onDrawerClose
) {
  const unread = getUnread(mails);
  const renderSinglebucket = (bucket, key) => {
    const onClick = () => {
      filterAction({ bucket });
      if (onDrawerClose) {
        onDrawerClose();
      }
    };
    const selectedBucket = bucket === filterAttr.bucket;
    const activeClass = selectedBucket ? 'active' : '';
    const iconColor =
      bucketColor[buckets.findIndex(buckets => buckets === bucket)];
    return (
      <ListItem key={`bucket${key}`} onClick={onClick} className={activeClass}>
        <ListItemIcon>
          <Icon style={{ color: iconColor }}>{bucketIcon[bucket]}</Icon>
        </ListItemIcon>
        <ListItemText primary={bucket} />
        {unread[bucket] ? (
          <span className="counter">{unread[bucket]}</span>
        ) : (
          ''
        )}
      </ListItem>
    );
  };
  return (
    <List>
      {buckets.map((bucket, index) => renderSinglebucket(bucket, index))}
    </List>
  );
}
export { buckets, bucketColor };
