import styled from 'styled-components';
import { palette } from 'styled-theme';
import { transition, borderRadius } from '../../settings/style-util';
import WithDirection from '../../settings/withDirection';
import Buttons from '../uielements/button';
import IconButton from '../uielements/iconbutton/';
import Icon from '../uielements/icon';
import Rates from '../uielements/rate';
import Done from '../../images/done.svg';
import Radios, { RadioGroup as RadioGroups } from '../uielements/radio';
import Selects from '../uielements/select';
import TextFields from '../uielements/textfield';

const MicIcon = styled(Icon)``;
const CartIcon = styled(Icon)``;
const GridIcon = styled(Icon)``;
const ListIcon = styled(Icon)``;
const ClearIcon = styled(Icon)``;
const TextField = styled(TextFields)``;

const ClearIconButtton = styled(IconButton)``;

const WDSelect = styled(Selects)``;
const Radio = styled(Radios)`
  width: 42px;
  height: 42px;
`;

const RadioGroup = styled(RadioGroups)`
  > label {
    margin-right: 0;

    > span {
      &:last-child {
        color: ${palette('grey', 6)};
        font-size: 13px;
      }
    }
  }
`;

const SearchBoxWrapper = styled.div`
  position: relative;

  ${TextField} {
    width: 100%;

    input {
      width: calc(100% - 35px);
    }
  }

  ${ClearIcon} {
    font-size: 17px;
  }

  ${ClearIconButtton} {
    position: absolute;
    bottom: 0;
    right: 0;
    width: 32px;
    height: 32px;
    border-radius: 2px;
  }
`;

const WDButton = styled(Buttons)`
  line-height: 42px;
`;
const Rate = styled(Rates)``;

const SidebarItem = styled.div`
  width: 100%;
  display: flex;
  flex-direction: column;
  margin-bottom: 40px;
  border: 0;

  .algoliaSidebarTitle {
    font-size: 14px;
    font-weight: 500;
    color: ${palette('grey', 8)};
    line-height: 1.3;
    margin: 0 0 20px;
    display: flex;
  }

  &.noPaper {
    padding: 0;
    margin-bottom: 25px;
    background-color: transparent;
    box-shadow: none;
  }

  &.contentBox {
    * {
      box-sizing: content-box;
    }
  }

  &.inline {
    flex-direction: row;
    align-items: center;
    justify-content: space-between;

    .algoliaSidebarTitle {
      margin-bottom: 0;
    }
  }
`;

const WDSidebarWrapper = styled.div`
  width: 240px;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  box-sizing: border-box;
  margin: ${props =>
    props['data-rtl'] === 'rtl' ? '0 0 0 40px' : '0 40px 0 0'};
  ${transition()};

  * {
    box-sizing: border-box;
  }

  @media only screen and (max-width: 767px) {
    width: 100%;
    margin-left: ${props =>
      props['data-rtl'] === 'rtl' ? '30px' : 'calc(-100% + -30px)'};
    margin-right: ${props =>
      props['data-rtl'] === 'rtl' ? 'calc(-100% + -30px)' : '30px'};
  }

  @media only screen and (min-width: 768px) and (max-width: 1199px) {
    width: 270px;
  }

  /* rating */
  .ais-RatingMenu {
    .ais-RatingMenu-list {
      padding: 0;
      list-style: none;

      .ais-RatingMenu-item {
        margin-top: 6px;

        &:first-child {
          margin-top: 0;
        }

        .ais-RatingMenu-link {
          display: flex;
          align-items: center;
          text-decoration: none;

          .ais-RatingMenu-label {
            font-size: 13px;
            color: ${palette('indigo', 5)};
            margin: ${props =>
              props['data-rtl'] === 'rtl' ? '0 10px 0 0' : '0 0 0 10px'};
          }

          .ais-RatingMenu-count {
            color: ${palette('indigo', 5)};
            border-radius: 31px;
            background-color: rgba(39, 81, 175, 0.1);
            font-size: 12px;
            padding: 2px 10px;
            margin: ${props =>
              props['data-rtl'] === 'rtl' ? '0 auto 0 0' : '0 0 0 auto'};

            &:before,
            &:after {
              display: none;
            }
          }
        }
      }
    }
  }

  /* Range Slider */
  .alRangeSlider {
    width: 100%;
    display: flex;
    flex-direction: column;
    margin-top: -25px;
    padding: 0 6px;

    .alRangeNumber {
      display: flex;
      justify-content: flex-end;
      align-items: center;
      margin-bottom: 15px;
      color: ${palette('grey', 6)};

      span {
        font-size: 13px;
        font-weight: 500;
        color: ${palette('grey', 6)};
        line-height: 1.3;
        padding: ${props =>
          props['data-rtl'] === 'rtl' ? '0 0 0 10px' : '0 10px 0 0'};

        &:last-child {
          padding: ${props =>
            props['data-rtl'] === 'rtl' ? '0 10px 0 0' : '0 0 0 10px'};
        }
      }
    }

    .rheostat-horizontal {
      height: 13px;

      .rheostat-background {
        width: 100%;
        height: 3px;
        background-color: ${palette('grey', 2)};
        display: flex;
        margin-top: 5px;
      }

      .rheostat-progress {
        position: absolute;
        top: 5px;
        height: 3px;
        background-color: ${palette('indigo', 5)};
      }

      button {
        width: 13px;
        height: 13px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background-color: ${palette('indigo', 5)};
        border: 0px;
        outline: 0px;
        top: 0;
        padding: 0;
        transform: scale(1);
        margin-left: -6px;
        z-index: 1;
        cursor: pointer;
        transition: transform 0.35s;

        &:active {
          transform: scale(1.5);
        }
      }
    }
  }

  /* Checkbox */
  .ais-RefinementList__root {
    .ais-RefinementList__SearchBox {
      .ais-RefinementList__noResults {
        font-size: 15px;
        text-align: center;
        font-weight: 400;
        color: ${palette('grey', 8)};
        line-height: 1.3;
      }
    }

    .ais-RefinementList__item {
      margin-top: 10px;
      &:first-child {
        margin-top: 0;
      }

      label {
        display: flex;
        align-items: center;
        font-size: 13px;
        font-weight: 400;
        color: ${palette('grey', 6)};

        .ais-RefinementList__itemLabelSelected {
          color: ${palette('indigo', 5)};
          font-weight: 400;
        }

        .ais-RefinementList__itemCount {
          margin: ${props =>
            props['data-rtl'] === 'rtl' ? ' 0 auto 0 0' : '0 0 0 auto'};
        }

        .ais-RefinementList__itemBox {
          width: 18px;
          height: 18px;
          border-radius: 2px;
          cursor: pointer;
          background: none;
          box-shadow: inset 0 0 0 2px ${palette('grey', 7)};
          margin: ${props =>
            props['data-rtl'] === 'rtl' ? '0 0 0 15px' : '0 15px 0 0'};
        }
      }

      &:hover {
        .ais-RefinementList__itemBox {
          box-shadow: inset 0 0 0 2px ${palette('grey', 7)};
        }
      }

      &.ais-RefinementList__itemSelected {
        &:hover {
          .ais-RefinementList__itemBoxSelected {
            box-shadow: none;
          }
        }
        .ais-RefinementList__itemBoxSelected {
          box-shadow: none;
          color: #ffffff;
          background: ${palette('indigo', 5)} url(${Done});
          background-size: 16px 16px;
          background-repeat: no-repeat;
          background-position: center;
        }
      }
    }
  }

  /* Hirarchi */
  .ais-HierarchicalMenu__root {
    .ais-HierarchicalMenu__item {
      margin-top: 8px;
      float: ${props => (props['data-rtl'] === 'rtl' ? 'right' : 'left')};

      &:after {
        right: ${props => (props['data-rtl'] === 'rtl' ? 'inherit' : '-22px')};
        left: ${props => (props['data-rtl'] === 'rtl' ? '-22px' : 'inherit')};
        transform: ${props =>
          props['data-rtl'] === 'rtl'
            ? 'translateY(-50%) rotate(180deg)'
            : 'translateY(-50%) rotate(0deg)'};
      }

      &:first-child {
        margin-top: 0;
      }

      .ais-HierarchicalMenu__itemLink {
        .ais-HierarchicalMenu__itemLabel {
          font-size: 13px;
          color: ${palette('indigo', 5)};
        }

        .ais-HierarchicalMenu__itemCount {
          color: ${palette('indigo', 5)};
        }

        &:focus {
          text-decoration: none;
        }
      }

      &.ais-HierarchicalMenu__itemSelected {
        .ais-HierarchicalMenu__itemLabel {
          font-weight: 600;
        }

        .ais-HierarchicalMenu__itemCount {
          font-weight: 600;
        }
      }

      &.ais-HierarchicalMenu__itemParent {
        .ais-HierarchicalMenu__item {
          margin-top: 8px;
        }
      }
    }
  }

  .ais-ClearAll__root {
    padding: 12px 8px;
    display: block;
    font-weight: 700;
    border-radius: 2px;
    text-transform: uppercase;
    background-color: ${palette('indigo', 5)};
    box-shadow: ${palette('shadows', 1)};

    @media only screen and (max-width: 358px) {
      margin-top: 10px;
    }
  }
`;

const WDContentWrapper = styled.div`
  width: calc(100% - 280px);
  display: flex;
  flex-direction: column;
  box-sizing: border-box;

  * {
    box-sizing: border-box;
  }

  @media only screen and (max-width: 767px) {
    width: 100%;
    margin-right: ${props => (props['data-rtl'] === 'rtl' ? '30px' : '0')};
  }

  @media only screen and (min-width: 768px) and (max-width: 1199px) {
    width: calc(100% - 300px);
  }

  main {
    width: 100%;
    display: flex;
    flex-wrap: wrap;
    margin-bottom: 20px;
  }

  &.list {
    main {
      flex-direction: column;
    }
  }
`;

const WDGridListViewWrapper = styled.div`
  &.GridView {
    width: calc(100% / 3 - 14px);
    display: -webkit-box;
    display: -moz-box;
    display: -ms-flexbox;
    display: -webkit-flex;
    display: flex;
    flex-direction: column;
    padding: 0;
    background-color: #fff;
    margin: ${props =>
      props['data-rtl'] === 'rtl' ? '0 0 20px 20px' : '0 20px 20px 0'};
    position: relative;
    border: 0;
    box-shadow: ${palette('shadows', 1)};

    @media only screen and (min-width: 1025px) {
      &:nth-child(3n) {
        margin: ${props =>
          props['data-rtl'] === 'rtl' ? '0 0 20px 0' : '0 0 20px 0'};
      }
    }

    @media only screen and (min-width: 900px) and (max-width: 1024px) {
      width: calc(100% / 2 - 10px);
      flex-shrink: 0;
      margin: ${props =>
        props['data-rtl'] === 'rtl' ? '0 0 20px 20px' : '0 20px 20px 0'};

      &:nth-child(2n) {
        margin: ${props =>
          props['data-rtl'] === 'rtl' ? '0 0 20px 0' : '0 0 20px 0'};
      }
    }

    @media only screen and (min-width: 768px) and (max-width: 899px) {
      width: 100%;
      flex-shrink: 0;
    }

    @media only screen and (max-width: 767px) {
      width: calc(100% / 2 - 15px);
      margin: ${props =>
        props['data-rtl'] === 'rtl' ? '0 0 20px 20px' : '0 20px 20px 0'};

      &:nth-child(2n) {
        margin: ${props =>
          props['data-rtl'] === 'rtl' ? '0 0 20px 0' : '0 0 20px 0'};
      }
    }

    @media only screen and (max-width: 550px) {
      width: 100%;
      margin: 0 0 20px;
    }

    .alGridImage {
      width: 100%;
      height: 245px;
      display: -webkit-box;
      display: -moz-box;
      display: -ms-flexbox;
      display: -webkit-flex;
      display: flex;
      flex-shrink: 0;
      align-items: center;
      -webkit-align-items: center;
      justify-content: center;
      -webkit-justify-content: center;
      overflow: hidden;
      background-color: #ffffff;
      position: relative;

      &:after {
        content: '';
        width: 100%;
        height: 100%;
        display: flex;
        background-color: rgba(0, 0, 0, 0.8);
        position: absolute;
        top: 0;
        left: 0;
        opacity: 0;
        ${transition()};
      }

      ${WDButton} {
        background-color: ${palette('indigo', 5)};
        border-color: ${palette('indigo', 5)} !important;
        color: #fff;
        z-index: 1;
        position: absolute;
        height: 42px;
        opacity: 0;
        padding: 0 20px;
        transform: scale(0.8);
        border-radius: 2px;
        box-shadow: ${palette('shadows', 1)};
        ${transition()};

        ${CartIcon} {
          color: #ffffff;
          font-size: 16px;
          margin: ${props =>
            props['data-rtl'] === 'rtl' ? '0 0 0 10px' : '0 10px 0 0'};
        }

        &:hover {
          background-color: ${palette('indigo', 7)};
          border-color: ${palette('indigo', 7)} !important;
        }
      }

      img {
        max-width: 100%;
      }

      @media only screen and (max-width: 991px) {
        width: 100%;
        overflow: hidden;
      }
    }

    .alGridContents {
      width: 100%;
      padding: 20px 25px;
      display: -webkit-box;
      display: -moz-box;
      display: -ms-flexbox;
      display: -webkit-flex;
      display: flex;
      flex-direction: column;
      border-top: 1px solid ${palette('grey', 2)};

      .alGridName {
        text-overflow: ellipsis;
        overflow: hidden;
        white-space: nowrap;
        margin-bottom: 5px;

        .ais-Highlight__nonHighlighted {
          font-size: 14px;
          font-weight: 500;
          color: ${palette('grey', 7)};
          line-height: 1.3;
        }
      }

      .alGridPriceRating {
        display: flex;
        align-items: center;

        .alGridPrice {
          font-size: 14px;
          font-weight: 400;
          color: #ffffff;
          padding: 5px 10px;
          line-height: 1;
          position: absolute;
          top: 30px;
          right: -1px;
          background-color: ${palette('indigo', 5)};
          ${borderRadius('2px 0 0 2px')};
        }

        .alGridRating {
          display: none;
          .ant-rate {
            display: flex;
            .ant-rate-star {
              font-size: 9px;
              margin-right: 2px;
            }
          }
        }
      }

      .alGridDescription {
        display: none;

        .ais-Highlight__nonHighlighted {
          font-size: 13px;
          font-weight: 400;
          color: ${palette('grey', 8)};
          line-height: 1.5;
        }
      }
    }

    &:hover {
      .alGridImage {
        &:after {
          opacity: 1;
        }

        ${WDButton} {
          opacity: 1;
          transform: scale(1);
        }
      }
    }
  }

  &.ListView {
    width: 100%;
    display: flex;
    padding: 10px;
    background-color: #ffffff;
    margin-bottom: 15px;
    border: 0;
    box-shadow: ${palette('shadows', 1)};

    @media only screen and (max-width: 991px) {
      flex-direction: column;
    }

    .alGridImage {
      width: 240px;
      height: auto;
      display: flex;
      flex-shrink: 0;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      position: relative;

      &:after {
        content: '';
        width: 100%;
        height: 100%;
        display: flex;
        background-color: rgba(0, 0, 0, 0.6);
        position: absolute;
        top: 0;
        left: 0;
        opacity: 0;
        ${transition()};
      }

      ${WDButton} {
        background-color: ${palette('indigo', 5)};
        border-color: ${palette('indigo', 5)} !important;
        color: #fff;
        z-index: 1;
        position: absolute;
        height: 42px;
        opacity: 0;
        padding: 0 20px;
        transform: scale(0.8);
        box-shadow: ${palette('shadows', 1)};
        ${transition()};

        ${CartIcon} {
          color: #ffffff;
          font-size: 16px;
          margin: ${props =>
            props['data-rtl'] === 'rtl' ? '0 0 0 10px' : '0 10px 0 0'};
        }

        &:hover {
          background-color: ${palette('indigo', 5)};
          border-color: ${palette('indigo', 5)} !important;
        }

        &.ant-btn-loading {
          i:not(.anticon-loading) {
            margin: ${props =>
              props['data-rtl'] === 'rtl' ? '0 10px 0 0' : '0 0 0 10px'};
          }
        }
      }

      &:hover {
        &:after {
          opacity: 1;
        }

        ${WDButton} {
          opacity: 1;
          transform: scale(1);
        }
      }

      img {
        max-width: 100%;
      }

      @media only screen and (max-width: 991px) {
        width: 100%;
        height: auto;
        overflow: hidden;
      }
    }

    .alGridContents {
      width: 100%;
      padding: 30px 15px;
      padding-left: 30px;
      display: flex;
      flex-direction: column;

      @media only screen and (max-width: 991px) {
        margin-top: 15px;
      }

      .alGridName {
        .ais-Highlight__nonHighlighted {
          font-size: 16px;
          font-weight: 500;
          color: ${palette('grey', 8)};
          line-height: 1.3;
          margin-bottom: 10px;
          display: flex;
        }
      }

      .alGridPriceRating {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;

        .alGridPrice {
          font-size: 13px;
          font-weight: 500;
          color: ${palette('grey', 7)};
          line-height: 1;
        }

        .alGridRating {
          ${Rate} {
            display: flex;
            align-items: center;

            label {
              margin-right: 2px;
              i {
                font-size: 11px;
              }
            }
          }
        }
      }

      .alGridDescription {
        .ais-Highlight__nonHighlighted {
          font-size: 13px;
          font-weight: 400;
          color: ${palette('grey', 5)};
          line-height: 1.5;
        }
      }
    }
  }
`;

const WDViewChanger = styled.div`
  display: flex;
  align-items: center;

  @media only screen and (max-width: 500px) {
    margin: ${props =>
      props['data-rtl'] === 'rtl' ? '0 auto 0 0' : '0 0 0 auto'};
  }

  button {
    font-size: 16px;
    text-align: center;
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    outline: 0;
    padding: 0;
    border: 0;
    margin-left: 0;
    cursor: pointer;
    background-color: ${palette('grey', 1)};
    ${transition()};

    ${GridIcon}, ${ListIcon} {
      color: ${palette('grey', 8)};
      font-size: 21px;
      width: 100%;
      line-height: 38px;
      height: auto;
    }

    &.active {
      background-color: ${palette('indigo', 5)};
      border-color: ${palette('indigo', 5)};
      ${GridIcon}, ${ListIcon} {
        color: #ffffff;
      }

      &:hover {
        background-color: ${palette('indigo', 5)};
        border-color: ${palette('indigo', 5)};

        ${GridIcon}, ${ListIcon} {
          color: #ffffff;
        }
      }
    }

    &:first-child {
      margin-left: 0;
    }

    &:hover {
      background-color: #ffffff;
      border-color: #ffffff;

      ${GridIcon}, ${ListIcon} {
        color: ${palette('grey', 8)};
      }
    }
  }
`;

const WDTopbarWrapper = styled.div`
  width: 100%;
  display: flex;
  align-items: center;
  margin-bottom: 30px;
  flex-wrap: wrap;

  @media only screen and (max-width: 500px) {
    margin-bottom: 20px;
  }

  .ais-Stats {
    font-size: 14px;
    font-weight: 500;
    opacity: 1;
    color: ${palette('grey', 7)};

    @media only screen and (max-width: 500px) {
      width: 100%;
      margin-top: 40px;
      order: 1;
    }
  }

  .sortingOpt {
    margin-left: auto;
    display: flex;

    @media only screen and (max-width: 500px) {
      margin-left: 0;
      width: 100%;
    }

    ${WDSelect} {
      margin-left: ${props => (props['data-rtl'] === 'rtl' ? '30px' : '0')};
      margin-right: ${props => (props['data-rtl'] === 'rtl' ? '0' : '30px')};
      color: ${palette('grey', 8)};
      box-sizing: content-box;

      * {
        box-sizing: content-box;
      }
    }
  }
`;

const WDVoiceSearchWrapper = styled.div`
  width: 100%;
  display: flex;
  border: 0;

  div {
    width: 100%;
    display: flex;
    align-items: center;
  }

  button {
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    position: relative;
    outline: 0;
    cursor: pointer;
    padding: 0;
    background-color: transparent;
    border: 2px solid ${palette('grey', 3)};
    ${transition()};
    ${borderRadius('50%')};

    ${MicIcon} {
      font-size: 18px;
      color: ${palette('grey', 4)};
      margin: 0;
      width: auto;
      height: auto;
      -webkit-transition: color 0.3s cubic-bezier(0.215, 0.61, 0.355, 1);
      -moz-transition: color 0.3s cubic-bezier(0.215, 0.61, 0.355, 1);
      -ms-transition: color 0.3s cubic-bezier(0.215, 0.61, 0.355, 1);
      -o-transition: color 0.3s cubic-bezier(0.215, 0.61, 0.355, 1);
      transition: color 0.3s cubic-bezier(0.215, 0.61, 0.355, 1);
    }

    &:hover {
      border-color: ${palette('indigo', 5)};

      ${MicIcon} {
        color: ${palette('indigo', 5)};
      }
    }
  }

  span {
    font-size: 14px;
    font-weight: 400;
    color: ${palette('grey', 6)};
    line-height: 1.4;
    margin: ${props =>
      props['data-rtl'] === 'rtl' ? '0 10px 0 0' : '0 0 0 10px'};
  }

  .voiceSearchRunning {
    button {
      border-color: ${palette('indigo', 5)};

      ${MicIcon} {
        color: ${palette('indigo', 5)};
      }
    }
  }
`;

const WDFooterWrapper = styled.footer`
  display: flex;
  width: 100%;
  align-items: center;
  justify-content: flex-end;
  margin-top: 40px;

  span {
    font-size: 13px;
    font-weight: 600;
    color: ${palette('grey', 5)};
    line-height: 1.3;
    margin: ${props =>
      props['data-rtl'] === 'rtl' ? '0 0 0 20px' : '0 20px 0 0'};
  }

  .logoWrapper {
    max-width: 150px;

    img {
      max-width: 100%;
    }
  }
`;

const LoaderElement = styled.div`
  width: 100%;
  height: 80vh;
  display: flex;
  align-items: center;
  justify-content: center;

  .loaderElement {
    height: 3em;
    width: 3em;
    animation: rotation 1s infinite linear;
    border: 2px solid rgba(51, 105, 231, 0.3);
    border-top-color: rgb(51, 105, 231);
    border-radius: 50%;
  }

  @keyframes rotation {
    to {
      transform: rotate(360deg);
    }
  }
`;

const PaginationStyleWrapper = styled.div`
  margin: 35px 0 0;

  .ais-Pagination__root {
    background-color: ${palette('grey', 1)};
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    padding: 0px;
    margin: 0;
    border: 0;
    box-shadow: none;

    .ais-Pagination__item {
      height: 28px;
      line-height: 28px;
      margin-right: 2px;
      padding: 0;

      &:last-child {
        margin-right: 0;
      }

      a {
        text-decoration: none;
        color: ${palette('grey', 8)};
        font-weight: 500;
      }

      &.ais-Pagination__itemSelected {
        border-radius: 2px;
        background-color: ${palette('indigo', 5)};

        a {
          color: #ffffff;
        }

        &:hover {
          background-color: ${palette('indigo', 5)};
          a {
            color: #ffffff;
          }
        }
      }

      &.ais-Pagination__itemNext,
      &.ais-Pagination__itemPrevious {
        .ais-Pagination__itemLink {
          color: ${palette('grey', 8)};
          font-size: 18px;
        }
      }
    }
  }
`;

const SidebarWrapper = WithDirection(WDSidebarWrapper);
const ContentWrapper = WithDirection(WDContentWrapper);
const Button = WithDirection(WDButton);
const VoiceSearch = WithDirection(WDVoiceSearchWrapper);
const GridListViewWrapper = WithDirection(WDGridListViewWrapper);
const TopbarWrapper = WithDirection(WDTopbarWrapper);
const FooterWrapper = WithDirection(WDFooterWrapper);
const ViewChanger = WithDirection(WDViewChanger);
const Select = WithDirection(WDSelect);

export {
  SidebarWrapper,
  SidebarItem,
  MicIcon,
  CartIcon,
  GridIcon,
  ListIcon,
  Button,
  Rate,
  SearchBoxWrapper,
  ClearIcon,
  ClearIconButtton,
  Select,
  ViewChanger,
  ContentWrapper,
  FooterWrapper,
  GridListViewWrapper,
  VoiceSearch,
  LoaderElement,
  TopbarWrapper,
  TextField,
  Radio,
  RadioGroup,
  PaginationStyleWrapper,
};
