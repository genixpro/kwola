import styled from 'styled-components';
import Paper from '../uielements/paper';
import Icons from '../uielements/icon/index.js';

const Icon = styled(Icons)`
  padding-right: 15px;
`;

const NotificationWrapper = styled(Paper)`
  flex-grow: 0;
  max-width: 580px;
  background-color: #323232 !important;
  color: #fff;
  display: flex;
  padding: 15px 35px 15px 24px;
  flex-wrap: wrap;
  align-items: center;
  pointer-events: initial;

  svg {
    padding-right: 17px;
    width: 21px;
    height: 21px;
  }

  .msgTxt {
    font-size: 14px;
    color: #fff;
    font-weight: 400;
  }

  .msgDesc {
    width: 100%;
    font-size: 13px;
    color: #fff;
    font-weight: 400;
    margin-top: 15px;
  }

  button {
    margin-left: 10px;
    background-color: transparent;
    color: #9fa8da;
    font-weight: 700;
    line-height: 1.2;
    border: 0;
    outline: 0;
    box-shadow: none;
    text-transform: uppercase;
    padding: 0;
    font-size: 12px;
    cursor: pointer;
    transition: color 0.3s;

    &:hover {
      color: #c5cae9;
    }
  }
`;

export { Icon };
export default NotificationWrapper;
