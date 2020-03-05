import styled from 'styled-components';
import { palette } from 'styled-theme';
import Icons from '../../components/uielements/icon';
import Buttons from '../../components/uielements/button';
import { transition, borderRadius } from '../../settings/style-util';
import IconButtons from '../../components/uielements/iconbutton';

const Button = styled(Buttons)``;

const Icon = styled(Icons)`
  font-size: 21px;
  color: ${palette('indigo', 3)};
`;

const CloseButton = styled(IconButtons)`
  width: 28px;
  height: 28px;
  position: absolute;
  top: 0;
  right: 0;

  ${Icon} {
    font-size: 14px;
    color: ${palette('grey', 4)};
    transition: all 0.25s ease;
  }

  &:hover {
    ${Icon} {
      color: ${palette('grey', 8)};
    }
  }
`;

const BreadCrumbSwitch = styled.div`
  padding: 10px 20px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  background-color: ${palette('grey', 2)};
  margin-bottom: 20px;

  h3 {
    font-size: 15px;
    font-weight: 500;
    color: ${palette('grey', 8)};
    line-height: 1;
    margin: 0;
  }
`;

const SwitcherBtns = styled.div`
  width: 100%;
  padding: 20px 20px 15px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  background-color: ${palette('grey', 2)};
  margin-bottom: 20px;

  h4 {
    font-size: 15px;
    font-weight: 500;
    color: ${palette('grey', 8)};
    line-height: 1;
    margin: 0 0 7px;
    padding: 0;
    text-transform: capitalize;
  }

  .themeSwitchBtnWrapper {
    width: 100%;
    display: flex;
    flex-flow: row wrap;
    align-items: center;

    ${Button} {
      width: 20px;
      height: 20px;
      display: flex;
      margin: ${props =>
        props['data-rtl'] === 'rtl' ? '5px 0 5px 10px' : '5px 10px 5px 0'};
      border: 1px solid #e4e4e4;
      outline: 0;
      padding: 0;
      min-width: auto;
      min-height: auto;
      background: none;
      justify-content: center;
      flex-shrink: 0;
      position: relative;
      cursor: pointer;
      ${borderRadius('3px')};

      &.languageSwitch {
        border: 0;
        width: 30px;
        height: auto;

        &.selectedTheme {
          &:after {
            bottom: -7px;
            left: ${props =>
              props['data-rtl'] === 'rtl' ? 'inherit' : '11px'};
            right: ${props =>
              props['data-rtl'] === 'rtl' ? '11px' : 'inherit'};
          }
        }
      }

      img {
        width: 100%;
      }

      &.selectedTheme {
        &:after {
          content: '';
          display: -webkit-inline-flex;
          display: -ms-inline-flex;
          display: inline-flex;
          width: 0;
          height: 0;
          border-style: solid;
          border-width: 0 4px 4px 4px;
          border-color: transparent transparent ${palette('indigo', 4)}
            transparent;
          position: absolute;
          bottom: -7px;
          left: ${props => (props['data-rtl'] === 'rtl' ? 'inherit' : '4px')};
          right: ${props => (props['data-rtl'] === 'rtl' ? '4px' : 'inherit')};
        }
      }
    }
  }
`;

const SwitcherBlock = styled.div`
  width: 100%;
  height: 100%;
  padding: 20px;
  overflow: hidden;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
`;

const ThemeSwitcherHeader = styled.div`
  padding: 25px 15px;
  width: 100%;
  background-color: #fff;
  border-bottom: 1px solid ${palette('grey', 2)};

  .componentTitle {
    font-size: 18px;
    font-weight: 500;
    color: ${palette('grey', 8)};
    line-height: 1;
    width: 100%;
    margin: 0;
    text-align: center;
    display: flex;
    justify-content: center;
  }
`;

const PurchaseActionBtn = styled.div`
  width: 100%;
  padding: 25px 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #ffffff;
  border-top: 1px solid ${palette('grey', 2)};

  .purchaseBtn {
    width: calc(100% - 50px);
    height: 42px;
    font-size: 14px;
    font-weight: 700;
    color: #fff;
    text-decoration: none;
    background-color: ${palette('indigo', 5)};
    text-transform: uppercase;
    line-height: 1;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    ${borderRadius('2px')};
    ${transition()};

    &:hover {
      background-color: ${palette('indigo', 8)};
    }
  }
`;

const SidebarInner = styled.div`
  height: 100%;
  display: flex;
  flex-direction: column;
  position: relative;
`;

export {
  BreadCrumbSwitch,
  SwitcherBtns,
  ThemeSwitcherHeader,
  SwitcherBlock,
  PurchaseActionBtn,
  Button,
  SidebarInner,
  Icon,
  CloseButton,
};
