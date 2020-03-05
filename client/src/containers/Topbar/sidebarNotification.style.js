import styled from 'styled-components';
import { palette } from 'styled-theme';
import { transition, borderRadius, boxShadow } from '../../settings/style-util';
import IconButton from '../../components/uielements/iconbutton';
import Icons from '../../components/uielements/icon';

const SidebarContent = styled.div`
  display: flex;
  flex-direction: column;
  background-color: #ffffff;
  margin: 0;
  width: 100%;
  height: calc(100% - 65px);
  min-width: 160px;
  flex-shrink: 0;
  ${borderRadius('2px')};
  ${boxShadow('0 2px 4px rgba(0,0,0,0.26)')};
  ${transition()};
  box-sizing: border-box;
  cursor: default;

  * {
    box-sizing: border-box;
  }

  @media only screen and (max-width: 767px) {
    width: auto;
  }

  .dropdownHeader {
    background-color: #ffffff;
    border-bottom: 1px solid ${palette('grey', 2)};
    margin-bottom: 0px;
    padding: 15px 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    ${borderRadius('2px 2px 0 0')};

    h3 {
      font-size: 14px;
      font-weight: 500;
      color: ${palette('text', 8)};
      text-align: center;
      margin: 0;
    }
  }

  .dropdownBody {
    height: 100%;
    display: flex;
    flex-direction: column;
    background-color: ${palette('grey', 1)};

    .scroll-content {
      padding: 10px;
    }

    a {
      text-decoration: none;
    }

    .dropdownListItem {
      padding: 20px;
      margin-bottom: 10px;
      flex-shrink: 0;
      text-decoration: none;
      display: flex;
      flex-direction: column;
      text-decoration: none;
      cursor: pointer;
      background-color: #fff;
      text-align: ${props => (props['data-rtl'] === 'rtl' ? 'right' : 'left')};
      ${boxShadow('0 0px 3px 0px rgba(0,0,0,0.2)')};
      ${transition()};

      &:last-child {
        margin-bottom: 0;
      }

      .listHead {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 5px;
      }

      h5 {
        font-size: 14px;
        font-weight: 500;
        color: ${palette('grey', 8)};
        margin: 0;
      }

      p {
        font-size: 12px;
        font-weight: 400;
        color: ${palette('grey', 6)};
        line-height: 1.5;
        overflow: hidden;
        margin: 10px 0 0;
      }

      .date {
        font-size: 11px;
        color: ${palette('grey', 8)};
        flex-shrink: 0;
      }
    }
  }

  .viewAllBtn {
    font-size: 13px;
    font-weight: 500;
    color: ${palette('grey', 7)};
    background-color: transparent;
    border-top: 1px solid ${palette('grey', 3)};
    padding: 15px 30px;
    display: flex;
    text-decoration: none;
    align-items: center;
    justify-content: center;
    text-align: center;
    cursor: pointer;
    ${transition()};

    &:hover {
      color: ${palette('indigo', 5)};
    }
  }

  .dropdownFooterLinks {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background-color: #ffffff;
    border-top: 1px solid ${palette('grey', 2)};
    padding: 15px 30px 15px 20px;

    a {
      font-size: 13px;
      font-weight: 500;
      color: ${palette('indigo', 5)};
      text-decoration: none;
      line-height: 1;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    h3 {
      font-size: 14px;
      font-weight: 500;
      color: ${palette('grey', 8)};
      line-height: 1.3;
      margin: 0;
    }
  }

  .noItemMsg {
    min-height: 300px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    font-weight: 300;
    color: ${palette('grayscale', 1)};
    line-height: 1.2;
  }

  &.withImg {
    .dropdownListItem {
      display: flex;
      flex-direction: row;
      .userImgWrapper {
        width: 35px;
        height: 35px;
        overflow: hidden;
        margin: ${props =>
          props['data-rtl'] === 'rtl' ? '0 0 0 15px' : '0 15px 0 0'};
        display: -webkit-inline-flex;
        display: -ms-inline-flex;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        background-color: ${palette('grayscale', 9)};
        ${borderRadius('50%')};
        img {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }
      }
      .listContent {
        width: 100%;
        display: flex;
        flex-direction: column;
        .listHead {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 10px;
        }
        h5 {
          margin-bottom: 0;
          padding: ${props =>
            props['data-rtl'] === 'rtl' ? '0 0 0 15px' : '0 15px 0 0'};
        }
        .date {
          font-size: 11px;
          flex-shrink: 0;
        }
        p {
          white-space: normal;
          line-height: 1.5;
          margin-top: 0;
        }
      }
    }
  }
`;

const Icon = styled(Icons)`
  font-size: 21px;
  color: ${palette('grey', 6)};
  margin-right: 25px;
  ${transition()};
`;

const CloseButton = styled(IconButton)`
  width: 23px;
  height: 23px;
  position: absolute;
  top: 5px;
  right: 5px;
  z-index: 1900;
  cursor: pointer;

  ${Icon} {
    font-size: 13px;
    color: ${palette('grey', 4)};
    transition: all 0.25s ease;
    margin: 0;
  }

  &:hover {
    ${Icon} {
      color: ${palette('grey', 0)};
    }
  }
`;

export { SidebarContent, Icon, CloseButton };
