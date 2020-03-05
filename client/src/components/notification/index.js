import React from 'react';
import Notification from 'rc-notification';
import 'rc-notification/assets/index.css';
import NotificationWrapper, { Icon } from './style';
import './notification.css';

let notification = null;
Notification.newInstance(
  {
    style: { top: 60, right: 50, transform: 'translateX(0px)' },
  },
  n => (notification = n)
);
const createNotification = (
  type = 'info',
  message = '',
  description = '',
  button = false,
  buttonText = 'undo'
) => {
  const Icons = {
    warning: 'warning',
    error: 'error',
    success: 'success',
    info: 'info',
  };
  const options = {
    content: (
      <NotificationWrapper className="notificationContent">
        {Icons[type] ? <Icon>{Icons[type]}</Icon> : ''}
        <span className="msgTxt">{message}</span>
        {description ? <span className="msgDesc">{description}</span> : ''}
        {button ? <button>{buttonText}</button> : ''}
      </NotificationWrapper>
    ),
    style: {},
    closable: true,
    duration: 4,
  };
  notification.notice(options);
};
export default createNotification;
