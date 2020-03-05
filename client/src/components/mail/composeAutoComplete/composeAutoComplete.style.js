import styled from 'styled-components';
import { palette } from 'styled-theme';
import { transition, borderRadius } from '../../../settings/style-util';

const AdressTag = styled.div``;

const ComposeAutoCompleteStyleWrapper = styled.div`
  .ReactTags__ {
    &tags {
      min-height: 44px;
      width: 100%;
      position: relative;
      margin-bottom: 0px;
      display: flex;
      align-items: center;
      padding: 0 20px;
      border-bottom: 1px solid ${palette('grey', 3)};
      ${borderRadius('0')};
    }

    &selected {
      width: 100%;
      padding: 4px 0;
      display: flex;
      flex-wrap: wrap;
    }

    &tag {
      font-size: 13px;
      font-weight: 400;
      line-height: 1;
      color: ${palette('grey', 8)};
      display: block;
      align-items: center;
      padding: 0 27px 0 10px;
      margin: 4px 6px 5px 0px;
      margin-right: ${props => (props['data-rtl'] === 'rtl' ? '0' : '5px')};
      margin-left: ${props => (props['data-rtl'] === 'rtl' ? '5px' : '0')};
      text-align: ${props => (props['data-rtl'] === 'rtl' ? 'right' : 'left')};
      height: 26px;
      line-height: 24px;
      word-break: break-word;
      background-color: ${palette('grey', 2)};
      border: 1px solid #ffffff;
      max-width: 300px;
      white-space: nowrap;
      position: relative;
      overflow: hidden;
      text-overflow: ellipsis;
      ${borderRadius('14px')};
    }

    &remove {
      background: none;
      background-color: transparent;
      border: 0;
      outline: 0;
      color: ${palette('grey', 6)};
      padding: 0;
      margin-right: ${props => (props['data-rtl'] === 'rtl' ? '8px' : '0')};
      margin-left: ${props => (props['data-rtl'] === 'rtl' ? '0' : '8px')};
      line-height: 22px;
      font-size: 18px;
      font-weight: 400;
      cursor: pointer;
      position: absolute;
      top: 1px;
      right: 9px;

      &:hover {
        color: ${palette('grey', 9)};
      }
    }

    &tagInput {
      width: 100%;
      display: inline-flex;
      flex: 1;
      align-items: center;
      min-height: 35px;

      &Field {
        font-size: 13px;
        font-weight: 400;
        color: ${palette('grey', 6)};
        line-height: inherit;
        text-align: ${props =>
          props['data-rtl'] === 'rtl' ? 'right' : 'left'};
        height: 100%;
        width: 100%;
        min-width: 100px;
        padding: 0;
        border: 0;
        outline: 0 !important;
        overflow: hidden;
        background-color: transparent;

        &::-webkit-input-placeholder {
          text-align: ${props =>
            props['data-rtl'] === 'rtl' ? 'right' : 'left'};
          color: ${palette('grey', 6)};
        }

        &:-moz-placeholder {
          text-align: ${props =>
            props['data-rtl'] === 'rtl' ? 'right' : 'left'};
          color: ${palette('grey', 6)};
        }

        &::-moz-placeholder {
          text-align: ${props =>
            props['data-rtl'] === 'rtl' ? 'right' : 'left'};
          color: ${palette('grey', 6)};
        }
        &:-ms-input-placeholder {
          text-align: ${props =>
            props['data-rtl'] === 'rtl' ? 'right' : 'left'};
          color: ${palette('grey', 6)};
        }
      }
    }

    &suggestions {
      z-index: 999;
      display: -webkit-flex;
      display: -ms-flex;
      display: flex;
      flex-direction: column;
      background-color: #fff;
      margin: 3px 0 0;
      overflow: hidden;
      word-break: break-word;
      border: 0;
      text-align: ${props => (props['data-rtl'] === 'rtl' ? 'right' : 'left')};
      position: absolute;
      right: ${props => (props['data-rtl'] === 'rtl' ? '20px' : 'auto')};
      left: ${props => (props['data-rtl'] === 'rtl' ? 'auto' : '20px')};
      top: 100%;
      ${borderRadius('2px')};
      box-shadow: ${palette('shadows', 1)};

      ul {
        list-style: none;
        padding: 0;
      }

      li {
        font-size: 13px;
        font-weight: 400;
        color: ${palette('grey', 8)};
        border-bottom: 0;
        line-height: 1.5;
        width: 100%;
        padding: 12px 20px;
        margin: 0;
        cursor: pointer;
        ${transition()};

        mark {
          font-weight: 700;
          color: ${palette('grey', 8)};
          background-color: transparent;
        }

        &:last-of-type {
          border-bottom: 0;
        }
      }
    }

    &activeSuggestion {
      background-color: rgba(0, 0, 0, 0.05);
    }
  }
`;

export { AdressTag };
export default ComposeAutoCompleteStyleWrapper;
